"""Knowledge-based early-warning system (KBEWS).

Seven digital-transformation failure rules are combined by certainty factors
and anchored to a gradient-boosting probability. The benchmark is 80
independent venture cohorts. Running this file regenerates both figures and
prints the paired results reported in the manuscript.

    python kbews.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 23
N_COHORTS = 80
N_VENTURES = 500
TEST_SIZE = 0.30
N_NOISE = 12
INNER_FOLDS = 3

SIGNAL = [
    "digital_leadership",
    "digital_capability",
    "financial_slack",
    "data_maturity",
    "transformation_ambition",
    "ecosystem_ties",
    "exploration",
    "exploitation",
    "ai_ambition",
    "business_model_adaptation",
    "firm_age",
    "firm_size",
]
SIGNAL_LABELS = [
    "Leadership",
    "Capability",
    "Financial slack",
    "Data maturity",
    "Ambition",
    "Ecosystem ties",
    "Exploration",
    "Exploitation",
    "AI ambition",
    "Adaptation",
    "Firm age",
    "Firm size",
]
NOISE = [f"noise_{i:02d}" for i in range(1, N_NOISE + 1)]
FEATURES = SIGNAL + NOISE
METHODS = ("Logistic regression", "Gradient boosting", "Knowledge base", "KBEWS")


@dataclass(frozen=True)
class Rule:
    code: str
    mechanism: str
    weight: float
    inputs: tuple[str, ...]

    def strength(self, frame: dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError


class CapabilityGap(Rule):
    def strength(self, frame):
        return _gap(frame["transformation_ambition"], frame["digital_capability"], 0.12)


class ResourceMisfit(Rule):
    def strength(self, frame):
        return _gap(frame["transformation_ambition"], frame["financial_slack"], 0.15)


class DataDeficit(Rule):
    def strength(self, frame):
        return _gap(frame["ai_ambition"], frame["data_maturity"], 0.12)


class AmbidexterityGap(Rule):
    def strength(self, frame):
        explore, exploit = frame["exploration"], frame["exploitation"]
        raw = (np.abs(explore - exploit) - 0.22) / 0.45
        return _mask(raw, explore, exploit)


class IsolatedAmbition(Rule):
    def strength(self, frame):
        ambition, ties = frame["transformation_ambition"], frame["ecosystem_ties"]
        raw = np.minimum((ambition - 0.58) / 0.30, (0.40 - ties) / 0.30)
        return _mask(raw, ambition, ties)


class LeadershipVoid(Rule):
    def strength(self, frame):
        ambition, lead = frame["transformation_ambition"], frame["digital_leadership"]
        raw = np.minimum((ambition - 0.55) / 0.30, (0.42 - lead) / 0.32)
        return _mask(raw, ambition, lead)


class WeakAdaptation(Rule):
    def strength(self, frame):
        ambition, adapt = frame["transformation_ambition"], frame["business_model_adaptation"]
        raw = np.minimum((ambition - 0.58) / 0.30, (0.38 - adapt) / 0.30)
        return _mask(raw, ambition, adapt)


def _gap(high: np.ndarray, low: np.ndarray, margin: float) -> np.ndarray:
    return _mask((high - low - margin) / 0.50, high, low)


def _mask(raw: np.ndarray, *parts: np.ndarray) -> np.ndarray:
    out = np.clip(raw, 0.0, 1.0)
    missing = np.zeros(out.shape, dtype=bool)
    for part in parts:
        missing |= np.isnan(part)
    out[missing] = np.nan
    return out


RULES: tuple[Rule, ...] = (
    CapabilityGap("R1", "Capability gap", 0.78, ("transformation_ambition", "digital_capability")),
    ResourceMisfit("R2", "Resource misfit", 0.70, ("transformation_ambition", "financial_slack")),
    DataDeficit("R3", "Data deficit", 0.76, ("ai_ambition", "data_maturity")),
    AmbidexterityGap("R4", "Ambidexterity imbalance", 0.58, ("exploration", "exploitation")),
    IsolatedAmbition("R5", "Ecosystem isolation", 0.52, ("transformation_ambition", "ecosystem_ties")),
    LeadershipVoid("R6", "Leadership void", 0.64, ("transformation_ambition", "digital_leadership")),
    WeakAdaptation("R7", "Weak adaptation", 0.48, ("transformation_ambition", "business_model_adaptation")),
)


def knowledge_matrix(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame = {name: x[:, i] for i, name in enumerate(FEATURES)}
    strengths = np.column_stack([rule.strength(frame) for rule in RULES])
    evidence = np.clip(np.nan_to_num(strengths, nan=0.0) * np.array([rule.weight for rule in RULES]), 0.0, 0.999)
    certainty = 1.0 - np.prod(1.0 - evidence, axis=1)
    return strengths, certainty


def generate(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """One cohort of ventures already pursuing a digital transformation.

    Latent constructs are Gaussian. The failure label is a stack of threshold
    mechanisms plus a size effect under high ambition that the rules do not
    encode. Indicators are those latents plus measurement noise, with 10% of
    signal values removed. Twelve irrelevant indicators are attached so a
    data-only model can overfit.
    """
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, 8))

    def unit(shift: float, scale: float, column: int) -> np.ndarray:
        return np.clip(shift + scale * latent[:, column], 0.0, 1.0)

    leadership = unit(0.52, 0.24, 0)
    capability = unit(0.50, 0.24, 1)
    slack = unit(0.50, 0.22, 2)
    maturity = unit(0.48, 0.24, 3)
    ambition = unit(0.54, 0.24, 4)
    ecosystem = unit(0.46, 0.22, 5)
    explore = unit(0.50, 0.20, 6)
    exploit = unit(0.50, 0.20, 7)
    ai_ambition = np.clip(0.75 * ambition + 0.25 * rng.normal(0.50, 0.18, n), 0.0, 1.0)
    adaptation = np.clip(rng.normal(0.46, 0.22, n), 0.0, 1.0)
    age = rng.uniform(0.0, 1.0, n)
    size = rng.uniform(0.0, 1.0, n)
    corner = ((ambition > 0.62) & (capability < 0.42) & (maturity < 0.48)).astype(float)
    eta = (
        -0.85
        + 2.80 * corner
        + 1.70 * ((ambition > 0.60) & (slack < 0.36)).astype(float)
        + 1.55 * ((ai_ambition > 0.64) & (maturity < 0.40)).astype(float)
        + 1.15 * (np.abs(explore - exploit) > 0.34).astype(float)
        + 1.20 * ((ambition > 0.62) & (ecosystem < 0.36)).astype(float)
        + 1.35 * ((ambition > 0.60) & (leadership < 0.38)).astype(float)
        - 1.25 * ((adaptation > 0.58) & (ambition > 0.50)).astype(float)
        + 1.10 * size * (ambition > 0.70)
        + rng.normal(0.0, 0.35, n)
    )
    label = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta)))

    def observe(values: np.ndarray, sd: float = 0.07) -> np.ndarray:
        return np.clip(values + rng.normal(0.0, sd, n), 0.0, 1.0)

    signal = np.column_stack(
        [
            observe(leadership),
            observe(capability),
            observe(slack),
            observe(maturity),
            observe(ambition),
            observe(ecosystem),
            observe(explore),
            observe(exploit),
            observe(ai_ambition),
            observe(adaptation, 0.05),
            age,
            size,
        ]
    )
    data = np.hstack([signal, rng.uniform(0.0, 1.0, size=(n, N_NOISE))])
    data[:, : len(SIGNAL)][rng.random(signal.shape) < 0.10] = np.nan
    return data.astype(float), label.astype(int)


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _booster(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=3,
        learning_rate=0.08,
        max_iter=160,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=seed,
    )


def _logistic(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=1.0, max_iter=500, random_state=seed)),
        ]
    )


def _anchor_features(probability: np.ndarray, certainty: np.ndarray, strengths: np.ndarray) -> np.ndarray:
    fired = np.nan_to_num(strengths, nan=0.0)
    return np.column_stack(
        [
            _logit(probability),
            certainty,
            certainty * (1.0 - probability),
            fired.max(axis=1),
            (fired >= 0.50).sum(axis=1),
        ]
    )


def recall_at_fpr(y_true: np.ndarray, scores: np.ndarray, target: float = 0.10) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    if target > fpr[-1]:
        return float(tpr[-1])
    return float(np.interp(target, fpr, tpr))


def _metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "auprc": float(average_precision_score(y_true, scores)),
        "auroc": float(roc_auc_score(y_true, scores)),
        "brier": float(brier_score_loss(y_true, np.clip(scores, 0.0, 1.0))),
        "recall10": recall_at_fpr(y_true, scores, 0.10),
    }


def _fit_cohort(x: np.ndarray, y: np.ndarray, seed: int) -> dict:
    train_idx, test_idx = next(
        StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed).split(x, y)
    )
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    strengths_train, certainty_train = knowledge_matrix(x_train)
    strengths_test, certainty_test = knowledge_matrix(x_test)

    logistic = _logistic(seed)
    logistic.fit(x_train, y_train)
    p_logistic = logistic.predict_proba(x_test)[:, 1]

    oof = np.zeros(len(y_train))
    for inner_train, inner_valid in StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed).split(x_train, y_train):
        model = _booster(seed)
        model.fit(x_train[inner_train], y_train[inner_train])
        oof[inner_valid] = model.predict_proba(x_train[inner_valid])[:, 1]
    anchor = LogisticRegression(C=1.0, max_iter=500, random_state=seed)
    anchor.fit(_anchor_features(oof, certainty_train, strengths_train), y_train)

    boosting = _booster(seed)
    boosting.fit(x_train, y_train)
    p_boost = boosting.predict_proba(x_test)[:, 1]
    p_kbews = anchor.predict_proba(_anchor_features(p_boost, certainty_test, strengths_test))[:, 1]
    scores = {
        "Logistic regression": p_logistic,
        "Gradient boosting": p_boost,
        "Knowledge base": certainty_test,
        "KBEWS": p_kbews,
    }
    return {
        "y": y_test,
        "scores": scores,
        "x": x_test,
        "strengths": strengths_test,
        "coefficients": np.r_[anchor.intercept_[0], anchor.coef_.ravel()],
        "metrics": {name: _metrics(y_test, score) for name, score in scores.items()},
    }


def _better_case(current: dict | None, cohort: dict) -> dict | None:
    y = cohort["y"]
    p_data = cohort["scores"]["Gradient boosting"]
    p_full = cohort["scores"]["KBEWS"]
    strengths = cohort["strengths"]
    caught = np.where((y == 1) & (p_data <= 0.40) & (p_full >= 0.55) & (p_full > p_data + 0.15))[0]
    for row in caught:
        fired = []
        inputs: set[int] = set()
        for index, rule in enumerate(RULES):
            strength = strengths[row, index]
            if not np.isfinite(strength) or strength < 0.50:
                continue
            fired.append({"code": rule.code, "mechanism": rule.mechanism, "strength": round(float(strength), 3)})
            for name in rule.inputs:
                inputs.add(SIGNAL.index(name))
        if len(fired) < 2:
            continue
        indicators = cohort["x"][row, : len(SIGNAL)]
        if not np.isfinite(indicators[list(inputs)]).all():
            continue
        lift = float(p_full[row] - p_data[row])
        if current is not None and lift <= current["lift"]:
            continue
        current = {
            "lift": round(lift, 3),
            "p_data": round(float(p_data[row]), 3),
            "p_kbews": round(float(p_full[row]), 3),
            "cf": round(float(cohort["scores"]["Knowledge base"][row]), 3),
            "fired": fired,
            "indicators": [None if not np.isfinite(value) else round(float(value), 3) for value in indicators],
            "highlight": sorted(inputs),
        }
    return current


def evaluate() -> dict:
    from scipy.stats import wilcoxon

    store = {name: {metric: [] for metric in ("auprc", "auroc", "brier", "recall10")} for name in METHODS}
    pooled = {name: [] for name in METHODS}
    labels: list[np.ndarray] = []
    coefficients = []
    failure_rates = []
    case = None
    for cohort_id in range(N_COHORTS):
        seed = SEED + cohort_id * 17
        x, y = generate(N_VENTURES, seed)
        failure_rates.append(float(y.mean()))
        fitted = _fit_cohort(x, y, seed)
        coefficients.append(fitted["coefficients"])
        labels.append(fitted["y"])
        for name in METHODS:
            for metric, value in fitted["metrics"][name].items():
                store[name][metric].append(value)
            pooled[name].append(fitted["scores"][name])
        case = _better_case(case, fitted)

    y_pool = np.concatenate(labels)
    score_pool = {name: np.concatenate(parts) for name, parts in pooled.items()}
    summary = {
        "cohorts": N_COHORTS,
        "ventures_per_cohort": N_VENTURES,
        "failure_rate_mean": round(float(np.mean(failure_rates)), 4),
        "coefficients_mean": np.mean(coefficients, axis=0).round(4).tolist(),
        "case": case,
        "means": {},
        "tests": [],
    }
    for name in METHODS:
        summary["means"][name] = {
            metric: round(float(np.mean(values)), 4) for metric, values in store[name].items()
        }
    proposed = store["KBEWS"]
    for baseline in ("Gradient boosting", "Logistic regression", "Knowledge base"):
        for metric in ("auprc", "brier", "recall10"):
            difference = np.asarray(proposed[metric]) - np.asarray(store[baseline][metric])
            p_value = float(wilcoxon(difference, alternative="two-sided", zero_method="wilcox").pvalue)
            summary["tests"].append(
                {
                    "baseline": baseline,
                    "metric": metric,
                    "baseline_mean": round(float(np.mean(store[baseline][metric])), 4),
                    "kbews_mean": round(float(np.mean(proposed[metric])), 4),
                    "mean_difference": round(float(np.mean(difference)), 4),
                    "p_value": p_value,
                }
            )
    summary["pooled_n"] = int(len(y_pool))
    return summary, y_pool, score_pool


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )


def draw_case(path: str, case: dict) -> None:
    _style()
    values = np.array([np.nan if item is None else item for item in case["indicators"]], dtype=float)
    highlight = set(case["highlight"])
    colors = ["#0072B2" if index in highlight else "#C5CDD6" for index in range(len(SIGNAL))]
    fig = plt.figure(figsize=(7.4, 4.6))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.7, 1.0), wspace=0.28)
    bars = fig.add_subplot(grid[0, 0])
    order = np.argsort(np.nan_to_num(values, nan=-1.0))
    ypos = np.arange(len(order))
    bars.barh(ypos, np.nan_to_num(values[order], nan=0.0), color=[colors[i] for i in order], edgecolor="#243447", linewidth=0.4, height=0.72)
    bars.set_yticks(ypos)
    bars.set_yticklabels([SIGNAL_LABELS[i] for i in order])
    bars.set_xlim(0, 1.05)
    bars.set_xlabel("Observed indicator")
    bars.spines["top"].set_visible(False)
    bars.spines["right"].set_visible(False)
    bars.axvline(0.5, color="#8A94A0", linewidth=0.6, linestyle=(0, (2, 2)))

    panel = fig.add_subplot(grid[0, 1])
    panel.set_xlim(0, 1)
    panel.set_ylim(-0.2, 3.4)
    panel.axis("off")
    rows = [
        (2.6, "Knowledge base", case["cf"], "#E69F00"),
        (1.6, "Gradient boosting", case["p_data"], "#009E73"),
        (0.6, "KBEWS", case["p_kbews"], "#0072B2"),
    ]
    for y, label, score, color in rows:
        panel.plot([0.02, 0.98], [y, y], color="#E4E8ED", linewidth=6, solid_capstyle="round")
        panel.plot([0.02, 0.02 + 0.96 * score], [y, y], color=color, linewidth=6, solid_capstyle="round")
        panel.text(0.02, y + 0.28, label, fontsize=8.5, color="#243447", ha="left")
        panel.text(0.98, y + 0.28, f"{score:.3f}", fontsize=8.5, color="#243447", ha="right")
    fired = ", ".join(item["code"] for item in case["fired"])
    panel.text(0.02, 0.02, f"Fired rules  {fired}", fontsize=8.5, color="#243447", ha="left")
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def draw_reliability(path: str, y_true: np.ndarray, scores: dict[str, np.ndarray]) -> None:
    _style()
    colors = {
        "Logistic regression": "#CC79A7",
        "Gradient boosting": "#009E73",
        "Knowledge base": "#E69F00",
        "KBEWS": "#0072B2",
    }
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.plot([0, 1], [0, 1], color="#8A94A0", linewidth=0.9, linestyle=(0, (3, 2)))
    edges = np.linspace(0.0, 1.0, 9)
    for name in METHODS:
        probability = np.clip(scores[name], 0.0, 1.0)
        centers = []
        observed = []
        for left, right in zip(edges[:-1], edges[1:]):
            chosen = (probability >= left) & (probability < right if right < 1 else probability <= right)
            if chosen.sum() < 40:
                continue
            centers.append(float(probability[chosen].mean()))
            observed.append(float(y_true[chosen].mean()))
        ax.plot(centers, observed, marker="o", markersize=4.5, linewidth=1.6, color=colors[name], label=name)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted failure probability")
    ax.set_ylabel("Observed failure frequency")
    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    summary, y_pool, score_pool = evaluate()
    if summary["case"] is None:
        raise RuntimeError("No illustrative failure case met the display rule.")
    draw_case("Figure_1.pdf", summary["case"])
    draw_reliability("Figure_2.pdf", y_pool, score_pool)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
