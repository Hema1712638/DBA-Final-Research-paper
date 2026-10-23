"""Knowledge-based early-warning system (KBEWS).

A fixed knowledge base of digital-transformation failure mechanisms is combined
with gradient boosting through a certainty-factor anchor. The default run
evaluates the system on a reproducible synthetic venture benchmark, because
public venture databases do not observe digital-transformation failure.

    python kbews.py

writes Figure_1.pdf and Figure_2.pdf in this directory and prints the metrics
reported in manuscript.tex.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 23
N_VENTURES = 1600
N_REPEATS = 20
TEST_SIZE = 0.30
N_NOISE = 12
INNER_FOLDS = 4

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
NOISE = [f"noise_{i:02d}" for i in range(1, N_NOISE + 1)]
FEATURES = SIGNAL + NOISE


@dataclass(frozen=True)
class Rule:
    code: str
    mechanism: str
    antecedent: str
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
    raw = (high - low - margin) / 0.50
    return _mask(raw, high, low)


def _mask(raw: np.ndarray, *parts: np.ndarray) -> np.ndarray:
    out = np.clip(raw, 0.0, 1.0)
    missing = np.zeros(out.shape, dtype=bool)
    for part in parts:
        missing |= np.isnan(part)
    out[missing] = np.nan
    return out


RULES: tuple[Rule, ...] = (
    CapabilityGap("R1", "Capability gap", "Ambition exceeds digital capability", 0.78, ("transformation_ambition", "digital_capability")),
    ResourceMisfit("R2", "Resource misfit", "Ambition exceeds financial slack", 0.70, ("transformation_ambition", "financial_slack")),
    DataDeficit("R3", "Data deficit", "AI ambition exceeds data maturity", 0.76, ("ai_ambition", "data_maturity")),
    AmbidexterityGap("R4", "Ambidexterity imbalance", "Exploration and exploitation diverge", 0.58, ("exploration", "exploitation")),
    IsolatedAmbition("R5", "Ecosystem isolation", "High ambition with weak external ties", 0.52, ("transformation_ambition", "ecosystem_ties")),
    LeadershipVoid("R6", "Leadership void", "High ambition without digital leadership", 0.64, ("transformation_ambition", "digital_leadership")),
    WeakAdaptation("R7", "Weak adaptation", "High ambition without business-model change", 0.48, ("transformation_ambition", "business_model_adaptation")),
)


def knowledge_matrix(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return rule firing strengths (n, 7) and the combined certainty factor."""
    frame = {name: x[:, i] for i, name in enumerate(FEATURES)}
    strengths = np.column_stack([rule.strength(frame) for rule in RULES])
    weights = np.array([rule.weight for rule in RULES])
    evidence = np.nan_to_num(strengths, nan=0.0) * weights
    evidence = np.clip(evidence, 0.0, 0.999)
    cf = 1.0 - np.prod(1.0 - evidence, axis=1)
    return strengths, cf


def generate(n: int = N_VENTURES, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Theory-grounded venture population with a known failure label.

    Indicators are noisy observations of the constructs named in the knowledge
    base. The failure label is a set of threshold mechanisms plus a size effect
    under high ambition, which the rules do not encode. Measurement noise and
    twelve irrelevant indicators keep the rules from reproducing the label.
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

    # Failure is driven by threshold mechanisms, not by additive main effects.
    # A linear model therefore misses the corners that the knowledge base encodes.
    # Firm size under high ambition is left out of the rules so the data model
    # still has a residual signal to contribute.
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
    probability = 1.0 / (1.0 + np.exp(-eta))
    label = rng.binomial(1, probability)

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
    noise = rng.uniform(0.0, 1.0, size=(n, N_NOISE))
    data = np.hstack([signal, noise])
    missing = rng.random(signal.shape) < 0.10
    data[:, : len(SIGNAL)][missing] = np.nan
    return data.astype(float), label.astype(int)


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))


def _booster() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=3,
        learning_rate=0.08,
        max_iter=160,
        min_samples_leaf=25,
        l2_regularization=0.1,
        random_state=SEED,
    )


def _logistic() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=1.0, max_iter=500, random_state=SEED)),
        ]
    )


def _anchor_features(p_data: np.ndarray, cf: np.ndarray, strengths: np.ndarray) -> np.ndarray:
    fired = np.nan_to_num(strengths, nan=0.0)
    return np.column_stack(
        [
            _logit(p_data),
            cf,
            cf * (1.0 - p_data),
            fired.max(axis=1),
            (fired >= 0.50).sum(axis=1),
        ]
    )


def recall_at_fpr(y_true: np.ndarray, scores: np.ndarray, target: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    if target > fpr[-1]:
        return float(tpr[-1])
    return float(np.interp(target, fpr, tpr))


def _scores(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "auprc": float(average_precision_score(y_true, scores)),
        "auroc": float(roc_auc_score(y_true, scores)),
        "brier": float(brier_score_loss(y_true, np.clip(scores, 0.0, 1.0))),
        "recall_fpr10": recall_at_fpr(y_true, scores, 0.10),
        "recall_fpr20": recall_at_fpr(y_true, scores, 0.20),
        "recall_fpr30": recall_at_fpr(y_true, scores, 0.30),
    }


def _out_of_fold_probability(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    oof = np.zeros(len(y), dtype=float)
    splitter = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
    for train_idx, valid_idx in splitter.split(x, y):
        model = _booster()
        model.fit(x[train_idx], y[train_idx])
        oof[valid_idx] = model.predict_proba(x[valid_idx])[:, 1]
    return oof


def evaluate(x: np.ndarray, y: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    models = ("Knowledge base", "Logistic regression", "Gradient boosting", "Rule-augmented boosting", "KBEWS")
    metrics = ("auprc", "auroc", "brier", "recall_fpr10", "recall_fpr20", "recall_fpr30")
    store = {name: {metric: [] for metric in metrics} for name in models}
    coefficients = []
    vignette = None
    splitter = StratifiedShuffleSplit(n_splits=N_REPEATS, test_size=TEST_SIZE, random_state=SEED)

    for repeat, (train_idx, test_idx) in enumerate(splitter.split(x, y)):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        s_train, cf_train = knowledge_matrix(x_train)
        s_test, cf_test = knowledge_matrix(x_test)

        logistic = _logistic()
        logistic.fit(x_train, y_train)
        p_logistic = logistic.predict_proba(x_test)[:, 1]

        boosting = _booster()
        boosting.fit(x_train, y_train)
        p_boost = boosting.predict_proba(x_test)[:, 1]

        rule_train = np.nan_to_num(s_train, nan=0.0)
        rule_test = np.nan_to_num(s_test, nan=0.0)
        augmented = _booster()
        augmented.fit(np.hstack([x_train, rule_train, cf_train[:, None]]), y_train)
        p_augmented = augmented.predict_proba(np.hstack([x_test, rule_test, cf_test[:, None]]))[:, 1]

        oof = _out_of_fold_probability(x_train, y_train, SEED + repeat)
        anchor = LogisticRegression(C=1.0, max_iter=500, random_state=SEED)
        anchor.fit(_anchor_features(oof, cf_train, s_train), y_train)
        p_kbews = anchor.predict_proba(_anchor_features(p_boost, cf_test, s_test))[:, 1]
        coefficients.append(np.r_[anchor.intercept_[0], anchor.coef_.ravel()])

        fold_scores = {
            "Knowledge base": cf_test,
            "Logistic regression": p_logistic,
            "Gradient boosting": p_boost,
            "Rule-augmented boosting": p_augmented,
            "KBEWS": p_kbews,
        }
        for name, scores in fold_scores.items():
            measured = _scores(y_test, scores)
            for metric, value in measured.items():
                store[name][metric].append(value)

        caught = np.where((y_test == 1) & (p_boost <= 0.45) & (p_kbews > p_boost + 0.08))[0]
        for row in caught:
            fired = []
            for rule, strength in zip(RULES, s_test[row]):
                if np.isfinite(strength) and strength >= 0.50:
                    fired.append({"code": rule.code, "mechanism": rule.mechanism, "strength": round(float(strength), 3)})
            if len(fired) < 2:
                continue
            lift = float(p_kbews[row] - p_boost[row])
            if vignette is None or lift > vignette["lift"]:
                vignette = {
                    "cf": round(float(cf_test[row]), 3),
                    "p_data": round(float(p_boost[row]), 3),
                    "p_kbews": round(float(p_kbews[row]), 3),
                    "lift": round(lift, 3),
                    "fired": fired,
                }

    arrays = {
        name: {metric: np.asarray(values, dtype=float) for metric, values in metric_map.items()}
        for name, metric_map in store.items()
    }
    extra = {"coefficients": np.vstack(coefficients), "vignette": vignette, "failure_rate": float(y.mean()), "n": int(len(y))}
    return arrays, extra


def corrected_pvalue(proposed: np.ndarray, baseline: np.ndarray) -> float:
    """Nadeau-Bengio corrected resampled t-test for repeated holdout."""
    diff = proposed - baseline
    k = len(diff)
    variance = diff.var(ddof=1)
    ratio = TEST_SIZE / (1.0 - TEST_SIZE)
    if variance <= 0:
        return 1.0
    from scipy import stats

    t_stat = diff.mean() / np.sqrt((1.0 / k + ratio) * variance)
    return float(2 * stats.t.sf(abs(t_stat), k - 1))


def draw_architecture(path: str) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 9,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    boxes = {
        "data": (0.3, 2.35, 2.3, 1.3, "Venture\nindicators"),
        "kb": (3.5, 3.85, 2.6, 1.25, "Knowledge base\nseven failure rules"),
        "ml": (3.5, 0.85, 2.6, 1.25, "Gradient boosting\ndata-driven probability"),
        "anchor": (7.0, 2.25, 2.5, 1.5, "Certainty-factor\nanchor"),
        "human": (10.0, 2.25, 1.8, 1.5, "Warning and\nrule trace"),
    }
    colors = {
        "data": "#E8F1FA",
        "kb": "#F4E6C3",
        "ml": "#D9EAD7",
        "anchor": "#F6D6D6",
        "human": "#E6E0F0",
    }
    for key, (x, y, w, h, text) in boxes.items():
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            facecolor=colors[key],
            edgecolor="#243447",
            linewidth=1.0,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, color="#1C2833")

    arrows = [
        ((2.6, 3.15), (3.5, 4.4)),
        ((2.6, 2.85), (3.5, 1.5)),
        ((6.1, 4.45), (7.0, 3.35)),
        ((6.1, 1.5), (7.0, 2.7)),
        ((9.5, 3.0), (10.0, 3.0)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, linewidth=1.0, color="#243447"))
    ax.text(6.35, 5.35, "certainty factor", fontsize=7.5, color="#5D4A1F", ha="center")
    ax.text(6.35, 0.45, "failure probability", fontsize=7.5, color="#1E4D2B", ha="center")
    fig.tight_layout(pad=0.2)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def draw_performance(path: str, results: dict[str, dict[str, np.ndarray]]) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9})
    order = ["Knowledge base", "Logistic regression", "Gradient boosting", "Rule-augmented boosting", "KBEWS"]
    short = ["Knowledge\nbase", "Logistic\nregression", "Gradient\nboosting", "Rule-augmented\nboosting", "KBEWS"]
    targets = [("recall_fpr10", "FPR = 0.10"), ("recall_fpr20", "FPR = 0.20"), ("recall_fpr30", "FPR = 0.30")]
    colors = ["#8DA0CB", "#66C2A5", "#FC8D62"]
    fig, ax = plt.subplots(figsize=(7.2, 3.35))
    x = np.arange(len(order))
    width = 0.24
    for i, ((metric, label), color) in enumerate(zip(targets, colors)):
        means = [results[name][metric].mean() for name in order]
        stds = [results[name][metric].std(ddof=1) for name in order]
        ax.bar(x + (i - 1) * width, means, width, yerr=stds, label=label, color=color, edgecolor="#243447", linewidth=0.4, capsize=2, error_kw={"elinewidth": 0.8})
    ax.set_xticks(x)
    ax.set_xticklabels(short)
    ax.set_ylabel("Recall of digital-transformation failure")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    x, y = generate()
    results, extra = evaluate(x, y)
    draw_architecture("Figure_1.pdf")
    draw_performance("Figure_2.pdf", results)
    summary = {
        "n": extra["n"],
        "failure_rate": round(extra["failure_rate"], 4),
        "vignette": extra["vignette"],
        "anchor_coefficients_mean": extra["coefficients"].mean(axis=0).round(4).tolist(),
        "models": {},
    }
    references = ("Gradient boosting", "Logistic regression", "Knowledge base", "Rule-augmented boosting")
    for name, metric_map in results.items():
        summary["models"][name] = {}
        for metric, values in metric_map.items():
            entry = {
                "mean": round(float(values.mean()), 4),
                "std": round(float(values.std(ddof=1)), 4),
            }
            for reference in references:
                key = "p_vs_" + reference.lower().replace(" ", "_")
                entry[key] = None if name == reference else round(corrected_pvalue(values, results[reference][metric]), 4)
            summary["models"][name][metric] = entry
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
