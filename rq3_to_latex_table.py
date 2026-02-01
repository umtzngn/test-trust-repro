"""
Generate LaTeX table for RQ3 from odds-ratio CSV.

Input:
  outputs/rq3_odds_ratios.csv

Output:
  outputs/rq3_logit_table.tex

Notes:
- Formats odds ratios and 95% CI
- Uses p-value stars (*, **, ***)
- Excludes intercept by default
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def p_stars(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def fmt(x, nd=2):
    if pd.isna(x):
        return ""
    return f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=str, default="outputs/rq3_odds_ratios.csv",
                    help="Input odds-ratio CSV from RQ3")
    ap.add_argument("--out", dest="out", type=str, default="outputs/rq3_logit_table.tex",
                    help="Output LaTeX table (.tex)")
    ap.add_argument("--drop-const", action="store_true", default=True,
                    help="Drop intercept (const) row")
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)

    if args.drop_const and "term" in df.columns:
        df = df[df["term"] != "const"].copy()

    # Keep and order terms explicitly (paper order)
    order = ["has_test_file", "log_lifetime", "is_ai"]
    df["order"] = df["term"].apply(lambda t: order.index(t) if t in order else 999)
    df = df.sort_values("order")

    # Build rows
    rows = []
    for _, r in df.iterrows():
        or_ci = f"{fmt(r['odds_ratio'])} [{fmt(r['ci_low'])}, {fmt(r['ci_high'])}]"
        stars = p_stars(r["p_value"])
        rows.append((r["term"], or_ci + stars))

    # Pretty names
    name_map = {
        "has_test_file": r"\texttt{has\_test\_file}",
        "log_lifetime": r"\texttt{log\_lifetime}",
        "is_ai": r"\texttt{is\_ai}",
    }

    # Write LaTeX
    with open(out, "w", encoding="utf-8") as f:
        f.write(r"\begin{table}[h]" + "\n")
        f.write(r"  \centering" + "\n")
        f.write(r"  \caption{RQ3: Logistic regression for merge probability (reduced model).}" + "\n")
        f.write(r"  \label{tab:rq3-logit}" + "\n")
        f.write(r"  \footnotesize" + "\n")
        f.write(r"  \setlength{\tabcolsep}{6pt}" + "\n")
        f.write(r"  \begin{tabular}{lr}" + "\n")
        f.write(r"    \toprule" + "\n")
        f.write(r"    Predictor & Odds ratio [95\% CI] \\" + "\n")
        f.write(r"    \midrule" + "\n")

        for term, val in rows:
            label = name_map.get(term, term)
            f.write(f"    {label} & {val} \\\\\n")

        f.write(r"    \midrule" + "\n")
        f.write(r"    \multicolumn{2}{l}{\emph{Notes:} Odds ratios reported. $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.}" + "\n")
        f.write(r"    \\" + "\n")
        f.write(r"  \end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")

    print("Saved LaTeX table to:", out)


if __name__ == "__main__":
    main()
