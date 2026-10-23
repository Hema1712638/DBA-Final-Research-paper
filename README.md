# KBEWS

Knowledge-based early-warning system for digital-transformation failure in entrepreneurial firms. A short communication for *Knowledge-Based Systems*, with the experiment that produces its two figures and two tables.

The benchmark is synthetic. Public venture files do not record digital-transformation failure. The numbers in `manuscript.tex` are holdout scores on that generator. They are not estimates for real firms.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python kbews.py
```

`kbews.py` rebuilds 80 independent cohorts, writes `Figure_1.pdf` and `Figure_2.pdf`, and prints the paired results used in the manuscript. Seed 23 is fixed.

Compile the manuscript with:

```bash
pdflatex manuscript.tex
pdflatex manuscript.tex
```

## What is in the repository

| File | Role |
| --- | --- |
| `kbews.py` | Knowledge base, anchor, benchmark, evaluation, figures |
| `manuscript.tex` | Short communication in Elsevier `elsarticle` form |
| `highlights.txt` | Separate highlights file required at submission |
| `Figure_1.pdf` | System diagram |
| `Figure_2.pdf` | Recall at three false-positive rates |
| `requirements.txt` | Pinned scientific Python stack |

## Journal fit

This is a regular short communication (at most 10 double-spaced manuscript pages, including tables and figures). It is not a special-issue proposal. Special-issue proposals go to the Editor-in-Chief. This manuscript goes through the journal's normal submission system as article type **Short communication**.

Before submission, replace the affiliation and postal address, confirm the corresponding email, and re-read every number against a fresh run of `kbews.py`. The generative-AI declaration names Cursor (Grok 4.7). Leave it in place, and review the text so the declaration stays true.
