# Test Trust Signal — Reproducibility Package

This repository provides the data and analysis code used in the paper:

**The Test Trust Signal: An Empirical Analysis of Testing Behavior in AI-Authored Pull Requests and Its Impact on Human Review and Integration Outcomes**

The purpose of this repository is to enable full reproducibility of all empirical results reported in the paper, covering Research Questions RQ1–RQ3.

---

## Repository Structure

```
.
├── data/
│   └── derived_data/
│       └── combined_model_df.parquet
│
├── src/
│   ├── rq1_test_inclusion.py
│   ├── rq2_test_comment_rates.py
│   ├── rq3_logit_merge_probability.py
│   └── rq3_to_latex_table.py
│
├── outputs/
│   ├── rq1_table.csv
│   ├── rq2_table.csv
│   ├── rq3_odds_ratios.csv
│   └── rq3_logit_table.tex
│
├── requirements.txt
└── README.md
```

---

## Data

All analyses are based on a single derived, analysis-ready dataset:

```
data/derived_data/combined_model_df.parquet
```

This dataset contains pull-request–level features derived from the AIDev dataset and augmented GitHub API data, including:
- PR source (AI vs. Human)
- Test-file indicators
- Test-related comment indicators
- Merge outcome
- Pull request lifetime
- Repository identifiers (used for clustered standard errors)

Raw GitHub API responses and intermediate processing artifacts are not required to reproduce the results and are therefore not included.

---

## Environment Setup

We recommend running the analysis in a clean Python virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Reproducing the Results

All commands should be executed from the repository root directory.

### RQ1 — Test-file inclusion in AI vs. human PRs

```bash
python src/rq1_test_inclusion.py \
  --data data/derived_data/combined_model_df.parquet \
  --out outputs/rq1_table.csv
```

---

### RQ2 — Test-related review discussion in AI PRs

```bash
python src/rq2_test_comment_rates.py \
  --data data/derived_data/combined_model_df.parquet \
  --out outputs/rq2_table.csv
```

---

### RQ3 — Logistic regression for merge probability

```bash
python src/rq3_logit_merge_probability.py \
  --data data/derived_data/combined_model_df.parquet \
  --out outputs/rq3_odds_ratios.csv
```

---

### LaTeX Table Generation (RQ3)

```bash
python src/rq3_to_latex_table.py \
  --in outputs/rq3_odds_ratios.csv \
  --out outputs/rq3_logit_table.tex
```

---

## Notes on Reported Values

- Differences reported in the paper as **percentage points (pp)** correspond to differences in proportions multiplied by 100.
- Confidence intervals for proportions use the **normal (Wald) approximation**.
- Logistic regression results report **odds ratios** with 95% confidence intervals.

---

## Ethics and Data Use

This repository contains no authentication tokens, personal data, or private repository content. All analyses are conducted on publicly available pull requests and derived aggregate features.

---

## Contact

For questions regarding the data or analysis, please contact the authors.
