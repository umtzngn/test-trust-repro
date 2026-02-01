"""
RQ2: Missing tests trigger more test-related review discussion (AI PRs only)

Computes:
- Among AI PRs with NO test files: proportion with >=1 test-related comment
- Among AI PRs with test files: proportion with >=1 test-related comment
- Wald (normal-approx) 95% CI for each proportion
- Difference (no tests - has tests) + two-proportion z-test + 95% CI

Input (parquet):
  data/derived_data/combined_model_df.parquet

Required columns:
  - source (must include "AI")
  - has_test_file (bool or 0/1)
  - has_test_comment (bool or 0/1)  # keyword-based proxy at PR level
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

    # Basic type normalization
    df["source"] = df["source"].astype(str)
    df["has_test_file"] = df["has_test_file"].fillna(False).astype(bool)

    if "has_test_comment" not in df.columns:
        raise ValueError(
            "RQ2 requires column 'has_test_comment' in combined_model_df.parquet "
            "(PR-level flag: at least one test-related comment)."
        )
    df["has_test_comment"] = df["has_test_comment"].fillna(False).astype(bool)

    return df


def compute_rq2(df: pd.DataFrame) -> pd.DataFrame:
    ai = df[df["source"] == "AI"].copy()
    if ai.empty:
        raise ValueError("No rows where source == 'AI'. Check your dataset.")

    rows = []
    for label, mask in [("No tests", ai["has_test_file"] == False),
                        ("Has tests", ai["has_test_file"] == True)]:
        g = ai[mask]
        n = int(len(g))
        x = int(g["has_test_comment"].sum())
        p = x / n if n else np.nan
        ci_low, ci_high = wald_ci(p, n) if n else (np.nan, np.nan)
        rows.append(
            {
                "Group": label,
                "n_prs": n,
                "n_with_test_comment": x,
                "proportion": p,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

    out = pd.DataFrame(rows)

    # Difference: (no tests) - (has tests)
    no = out[out["Group"] == "No tests"].iloc[0]
    ha = out[out["Group"] == "Has tests"].iloc[0]

    z, pval, d_low, d_high = two_prop_test(
        int(no["n_with_test_comment"]), int(no["n_prs"]),
        int(ha["n_with_test_comment"]), int(ha["n_prs"])
    )

    diff_row = pd.DataFrame([{
        "Group": "Diff (no-has)",
        "n_prs": np.nan,
        "n_with_test_comment": np.nan,
        "proportion": float(no["proportion"] - ha["proportion"]),
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
    ap.add_argument("--out", type=str, default="outputs/rq2_table.csv",
                    help="Output CSV path")
    args = ap.parse_args()

    data_path = Path(args.data)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_df(data_path)
    rq2_table = compute_rq2(df)

    rq2_table.to_csv(out_path, index=False)
    print("Saved:", out_path)

    # quick sanity print
    no = rq2_table[rq2_table["Group"] == "No tests"].iloc[0]
    ha = rq2_table[rq2_table["Group"] == "Has tests"].iloc[0]
    print("No tests proportion:", float(no["proportion"]))
    print("Has tests proportion:", float(ha["proportion"]))
    print("Diff proportion:", float(rq2_table[rq2_table["Group"] == "Diff (no-has)"].iloc[0]["proportion"]))


if __name__ == "__main__":
    main()
