"""
RQ1: Test-file inclusion (AI vs Human)

Computes:
- n PRs per group
- n PRs with has_test_file=True
- proportion + Wald (normal-approx) 95% CI
- difference in proportions (AI - Human) + two-proportion z-test + 95% CI

Input (parquet):
  data/derived_data/combined_model_df.parquet

Required columns:
  - source  (values include "AI" and "Human")
  - has_test_file (bool or 0/1)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep

Z_975 = 1.959963984540054  # 97.5th percentile of N(0,1)


def wald_ci(p: float, n: int) -> tuple[float, float]:
    """Wald (normal-approx) CI for a single proportion."""
    se = np.sqrt(p * (1 - p) / n) if n > 0 else np.nan
    return (p - Z_975 * se, p + Z_975 * se)


def two_prop_test(x1: int, n1: int, x2: int, n2: int, alpha: float = 0.05):
    """Two-proportion z-test + CI for difference in proportions (Wald)."""
    stat, pval = proportions_ztest([x1, x2], [n1, n2])
    ci_low, ci_high = confint_proportions_2indep(
        count1=x1,
        nobs1=n1,
        count2=x2,
        nobs2=n2,
        method="wald",
        compare="diff",
        alpha=alpha,
    )
    return float(stat), float(pval), float(ci_low), float(ci_high)


def load_df(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["source"] = df["source"].astype(str)
    df["has_test_file"] = df["has_test_file"].fillna(False).astype(bool)
    return df


def compute_rq1(df: pd.DataFrame) -> pd.DataFrame:
    rq1_df = df[df["source"].isin(["AI", "Human"])].copy()

    rows = []
    for grp in ["AI", "Human"]:
        g = rq1_df[rq1_df["source"] == grp]
        n = int(len(g))
        x = int(g["has_test_file"].sum())
        p = x / n if n else np.nan
        ci_low, ci_high = wald_ci(p, n) if n else (np.nan, np.nan)
        rows.append(
            {
                "Source": grp,
                "n_prs": n,
                "n_with_tests": x,
                "proportion": p,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

    out = pd.DataFrame(rows)

    # Difference (AI - Human)
    ai = out[out["Source"] == "AI"].iloc[0]
    hu = out[out["Source"] == "Human"].iloc[0]
    z, pval, d_low, d_high = two_prop_test(
        int(ai["n_with_tests"]), int(ai["n_prs"]),
        int(hu["n_with_tests"]), int(hu["n_prs"])
    )

    diff_row = pd.DataFrame([{
        "Source": "Diff (AI-Human)",
        "n_prs": np.nan,
        "n_with_tests": np.nan,
        "proportion": float(ai["proportion"] - hu["proportion"]),
        "ci_low": d_low,
        "ci_high": d_high,
        "z_stat": z,
        "p_value": pval,
    }])

    out["z_stat"] = np.nan
    out["p_value"] = np.nan
    out = pd.concat([out, diff_row], ignore_index=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/derived_data/combined_model_df.parquet",
                    help="Path to combined_model_df.parquet")
    ap.add_argument("--out", type=str, default="outputs/rq1_table.csv",
                    help="Output CSV path")
    args = ap.parse_args()

    data_path = Path(args.data)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_df(data_path)
    rq1_table = compute_rq1(df)

    rq1_table.to_csv(out_path, index=False)
    print("Saved:", out_path)

    # quick sanity print
    ai = rq1_table[rq1_table["Source"] == "AI"].iloc[0]
    hu = rq1_table[rq1_table["Source"] == "Human"].iloc[0]
    print("AI proportion:", float(ai["proportion"]))
    print("Human proportion:", float(hu["proportion"]))


if __name__ == "__main__":
    main()
