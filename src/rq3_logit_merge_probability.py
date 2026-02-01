"""
RQ3: Logistic regression for merge probability (reduced model)

Model:
  is_merged ~ has_test_file + log_lifetime + is_ai

This reduced model aligns with RQ1–RQ2 by focusing on test inclusion
and core control variables.

Method:
- statsmodels.Logit
- Cluster-robust standard errors by repo_id when available;
  otherwise HC1 heteroskedasticity-robust SE
- Reports odds ratios, 95% confidence intervals, and p-values
- Outputs saved to: outputs/rq3_odds_ratios.csv

Input (Parquet):
  data/derived_data/combined_model_df.parquet

Required columns:
  - is_merged (bool or 0/1)
  - has_test_file (bool or 0/1)
  - lifetime_days (used to compute log_lifetime if not present)
  - source (values include "AI" and "Human") OR precomputed is_ai

Derived variables:
  - is_ai (1 if source == "AI", else 0)
  - log_lifetime = log(lifetime_days + 1)

Optional:
  - repo_id (used for clustered standard errors)
"""

#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm

FEATURES = ["has_test_file", "log_lifetime", "is_ai"]

def load_df(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)

    # Keep only AI/Human like the notebook
    if "source" in df.columns:
        df["source"] = df["source"].astype(str)
        df = df[df["source"].isin(["AI", "Human"])].copy()

    # --- Match notebook preprocessing exactly ---
    # is_ai from source
    if "is_ai" not in df.columns:
        df["is_ai"] = (df["source"].str.upper() == "AI").astype(int)
    else:
        df["is_ai"] = pd.to_numeric(df["is_ai"], errors="coerce").fillna(0).astype(int)

    # log_lifetime = log1p(lifetime_days)
    df["lifetime_days"] = pd.to_numeric(df.get("lifetime_days"), errors="coerce")
    df["log_lifetime"] = np.log1p(df["lifetime_days"])

    # has_test_file bool -> int
    df["has_test_file"] = df.get("has_test_file").fillna(False).astype(int)

    # Outcome + cluster id
    df["is_merged"] = pd.to_numeric(df.get("is_merged"), errors="coerce")
    df["repo_id"] = pd.to_numeric(df.get("repo_id"), errors="coerce")

    return df

def fit_logit(df: pd.DataFrame):
    needed = FEATURES + ["is_merged", "repo_id"]
    model_df = df.dropna(subset=needed).copy()

    model_df["is_merged"] = model_df["is_merged"].astype(int)
    model_df["repo_id"] = model_df["repo_id"].astype(int)

    X = sm.add_constant(model_df[FEATURES], has_constant="add")
    y = model_df["is_merged"]

    res = sm.Logit(y, X).fit(
        disp=0,
        cov_type="cluster",
        cov_kwds={"groups": model_df["repo_id"]},
    )
    return res, model_df

def odds_ratio_table(res):
    params = res.params
    conf = res.conf_int()
    pvals = res.pvalues

    out = pd.DataFrame({
        "term": params.index,
        "odds_ratio": np.exp(params.values),
        "ci_low": np.exp(conf[0].values),
        "ci_high": np.exp(conf[1].values),
        "p_value": pvals.values,
    })
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/derived_data/combined_model_df.parquet")
    ap.add_argument("--out_csv", default="outputs/rq3_odds_ratios.csv")
    args = ap.parse_args()

    df = load_df(args.data)
    res, model_df = fit_logit(df)
    or_df = odds_ratio_table(res)

    or_df.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv}")
    print("McFadden pseudo R^2:", float(res.prsquared))
    print("N obs:", int(res.nobs))
    print("\nOdds ratios (selected):\n", or_df)

if __name__ == "__main__":
    main()
