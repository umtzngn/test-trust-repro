"""
RQ3: Logistic regression for merge probability (reduced model)

Model:
  is_merged ~ has_test_file + log_lifetime + is_ai

- Uses statsmodels.Logit
- Cluster-robust SE by repo_id if available, else HC1 robust SE
- Reports odds ratios + 95% CI + p-values
- Writes outputs/rq3_odds_ratios.csv

Input (parquet):
  data/derived_data/combined_model_df.parquet

Required columns:
  - is_merged (0/1 or bool)
  - has_test_file (bool or 0/1)
  - source (contains "AI" and "Human") OR is_ai column
  - lifetime_days OR log_lifetime

Optional:
  - repo_id (for clustered SE)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_df(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)

    # Normalize common fields
    if "is_merged" not in df.columns:
        raise ValueError("Missing required column: is_merged")
    df["is_merged"] = df["is_merged"].astype(int)

    if "has_test_file" not in df.columns:
        raise ValueError("Missing required column: has_test_file")
    df["has_test_file"] = df["has_test_file"].fillna(False).astype(int)

    # is_ai: use existing column if present; else infer from source
    if "is_ai" in df.columns:
        df["is_ai"] = df["is_ai"].fillna(0).astype(int)
    else:
        if "source" not in df.columns:
            raise ValueError("Need either 'is_ai' or 'source' to derive AI authorship.")
        df["source"] = df["source"].astype(str)
        df["is_ai"] = (df["source"] == "AI").astype(int)

    # log_lifetime: prefer existing; else build from lifetime_days
    if "log_lifetime" in df.columns:
        df["log_lifetime"] = pd.to_numeric(df["log_lifetime"], errors="coerce")
    else:
        if "lifetime_days" not in df.columns:
            raise ValueError("Need either 'log_lifetime' or 'lifetime_days'.")
        df["lifetime_days"] = pd.to_numeric(df["lifetime_days"], errors="coerce").fillna(0.0)
        df["log_lifetime"] = np.log1p(df["lifetime_days"])

    return df


def fit_logit(df: pd.DataFrame) -> sm.discrete.discrete_model.BinaryResultsWrapper:
    features = ["has_test_file", "log_lifetime", "is_ai"]

    model_df = df.dropna(subset=["is_merged", "has_test_file", "log_lifetime", "is_ai"]).copy()

    X = model_df[features].copy()
    X = sm.add_constant(X, has_constant="add")  # include intercept
    y = model_df["is_merged"].astype(int)

    # Robust SE choice
    cov_type = "HC1"
    cov_kwds = None
    if "repo_id" in model_df.columns and model_df["repo_id"].notna().any():
        cov_type = "cluster"
        cov_kwds = {"groups": model_df["repo_id"]}
        print("Using cluster-robust SE grouped by repo_id")
    else:
        print("repo_id not available -> using HC1 robust SE")

    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=0, cov_type=cov_type, cov_kwds=cov_kwds)
    return result


def odds_ratio_table(result: sm.discrete.discrete_model.BinaryResultsWrapper) -> pd.DataFrame:
    params = result.params
    conf = result.conf_int()
    pvals = result.pvalues

    out = pd.DataFrame({
        "term": params.index,
        "coef": params.values,
        "odds_ratio": np.exp(params.values),
        "ci_low": np.exp(conf[0].values),
        "ci_high": np.exp(conf[1].values),
        "p_value": pvals.values,
    })

    # Add McFadden pseudo R^2 (statsmodels: 1 - llf/llnull)
    try:
        out.attrs["mcfadden_pseudo_r2"] = float(result.prsquared)
    except Exception:
        out.attrs["mcfadden_pseudo_r2"] = np.nan

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/derived_data/combined_model_df.parquet",
                    help="Path to combined_model_df.parquet")
    ap.add_argument("--out", type=str, default="outputs/rq3_odds_ratios.csv",
                    help="Output CSV path")
    args = ap.parse_args()

    data_path = Path(args.data)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_df(data_path)
    res = fit_logit(df)

    print("\n=== RQ3 Logit Summary (reduced model) ===")
    print(res.summary())

    or_df = odds_ratio_table(res)
    or_df.to_csv(out_path, index=False)
    print("\nSaved:", out_path)

    # Print key interpretation numbers for sanity
    pseudo_r2 = or_df.attrs.get("mcfadden_pseudo_r2", np.nan)
    print("McFadden pseudo R^2:", pseudo_r2)

    # If you want: highlight main terms
    keep = ["const", "has_test_file", "log_lifetime", "is_ai"]
    print("\nOdds ratios (selected):")
    print(or_df[or_df["term"].isin(keep)][["term", "odds_ratio", "ci_low", "ci_high", "p_value"]])


if __name__ == "__main__":
    main()
