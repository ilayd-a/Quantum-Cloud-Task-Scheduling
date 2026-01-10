"""Analyze ablation study results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def extract_variant_name(output_path: str) -> str:
    """Extract variant tag from output path."""
    if "full_qubo" in output_path:
        return "Full QUBO"
    elif "no_balance" in output_path:
        return "No balance term"
    elif "low_penalty" in output_path:
        return "Low penalty"
    elif "high_penalty" in output_path:
        return "High penalty"
    elif "no_priority" in output_path:
        return "No priority bias"
    elif "shallow_p1" in output_path or "_p1" in output_path:
        return "p=1 (shallow)"
    elif "medium_p2" in output_path or "_p2" in output_path:
        return "p=2 (medium)"
    elif "deep_p4" in output_path or "_p4" in output_path:
        return "p=4 (deep)"
    return "Unknown"


def main():
    parser = argparse.ArgumentParser(description="Analyze ablation study results")
    parser.add_argument("--input-csv", type=Path, required=True, help="Input CSV from ablation study")
    parser.add_argument("--output-table", type=Path, help="Output LaTeX table")
    
    args = parser.parse_args()
    
    df = pd.read_csv(args.input_csv)
    
    # Extract variant from output_path or tag
    if "tag" in df.columns:
        df["variant"] = df["tag"].apply(extract_variant_name)
    else:
        df["variant"] = df["output_path"].apply(extract_variant_name)
    
    # Compute approximation ratio
    df["approximation_ratio"] = df["qaoa_makespan"] / df["classical_makespan"]
    
    # Group by variant
    ablation_results = df.groupby("variant").agg({
        "approximation_ratio": ["mean", "std", "count"],
        "relative_gap_pct": ["mean"],
    })
    
    ablation_results.columns = ['_'.join(col).strip() for col in ablation_results.columns.values]
    ablation_results = ablation_results.round(3)
    
    print("=" * 80)
    print("ABLATION STUDY RESULTS")
    print("=" * 80)
    print("\nApproximation Ratio by Variant:")
    print(ablation_results.to_string())
    
    # Create summary table
    summary = []
    for variant in ablation_results.index:
        mean_approx = ablation_results.loc[variant, "approximation_ratio_mean"]
        std_approx = ablation_results.loc[variant, "approximation_ratio_std"]
        count = int(ablation_results.loc[variant, "approximation_ratio_count"])
        summary.append({
            "Variant": variant,
            "Approx. Ratio": f"{mean_approx:.2f} ± {std_approx:.2f}",
            "N": count,
        })
    
    summary_df = pd.DataFrame(summary)
    print("\n" + "-" * 80)
    print("Summary Table:")
    print("-" * 80)
    print(summary_df.to_string(index=False))
    
    if args.output_table:
        from tabulate import tabulate
        table_tex = tabulate(summary_df, headers="keys", tablefmt="latex", showindex=False)
        args.output_table.parent.mkdir(parents=True, exist_ok=True)
        args.output_table.write_text(table_tex)
        print(f"\nLaTeX table saved to: {args.output_table}")


if __name__ == "__main__":
    main()

