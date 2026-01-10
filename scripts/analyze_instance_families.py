"""Analyze results by instance family (uniform, heavy-tailed, clustered)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def extract_instance_type(dataset_name: str) -> str | None:
    """Extract instance type from dataset name."""
    if "uniform" in dataset_name:
        return "uniform"
    elif "heavy_tailed" in dataset_name:
        return "heavy_tailed"
    elif "clustered" in dataset_name:
        return "clustered"
    return "default"


def load_results_from_csv(csv_path: Path) -> pd.DataFrame:
    """Load experiment results from CSV."""
    df = pd.read_csv(csv_path)
    
    # Extract instance type
    df["instance_type"] = df["dataset"].apply(extract_instance_type)
    
    # Compute approximation ratio
    df["approximation_ratio"] = df["qaoa_makespan"] / df["classical_makespan"]
    
    return df


def compute_family_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute statistics by instance family."""
    grouped = df.groupby("instance_type").agg({
        "approximation_ratio": ["mean", "std", "min", "max", "count"],
        "relative_gap_pct": ["mean", "std"],
        "qaoa_makespan": ["mean", "std"],
        "classical_makespan": ["mean"],
    })
    
    # Flatten column names
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    
    return grouped


def main():
    parser = argparse.ArgumentParser(description="Analyze results by instance family")
    parser.add_argument("--input-csv", type=Path, required=True, help="Input CSV with results")
    parser.add_argument("--output-csv", type=Path, help="Output CSV with family statistics")
    parser.add_argument("--output-table", type=Path, help="Output LaTeX table")
    
    args = parser.parse_args()
    
    df = load_results_from_csv(args.input_csv)
    
    if "instance_type" not in df.columns or df["instance_type"].isna().all():
        print("Warning: No instance type information found in dataset names")
        print("Expected patterns: 'uniform', 'heavy_tailed', 'clustered'")
        return
    
    # Statistics by family
    stats = compute_family_statistics(df)
    
    print("=" * 80)
    print("INSTANCE FAMILY ANALYSIS")
    print("=" * 80)
    print("\nStatistics by Instance Type:")
    print(stats.to_string())
    
    # Summary by family
    print("\n" + "-" * 80)
    print("Summary by Instance Family:")
    print("-" * 80)
    for instance_type in ["uniform", "heavy_tailed", "clustered", "default"]:
        family_df = df[df["instance_type"] == instance_type]
        if len(family_df) > 0:
            avg_approx = family_df["approximation_ratio"].mean()
            std_approx = family_df["approximation_ratio"].std()
            print(f"{instance_type:15s}: Approximation ratio = {avg_approx:.3f} ± {std_approx:.3f} (n={len(family_df)})")
    
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        stats.to_csv(args.output_csv)
        print(f"\nStatistics saved to: {args.output_csv}")
    
    if args.output_table:
        from tabulate import tabulate
        table_tex = tabulate(stats, headers=stats.columns, tablefmt="latex", floatfmt=".3f")
        args.output_table.parent.mkdir(parents=True, exist_ok=True)
        args.output_table.write_text(table_tex)
        print(f"LaTeX table saved to: {args.output_table}")


if __name__ == "__main__":
    main()

