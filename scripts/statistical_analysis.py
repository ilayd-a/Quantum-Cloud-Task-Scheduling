"""Statistical analysis tools for experiment results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def load_experiment_results(results_dir: Path) -> pd.DataFrame:
    """Load all experiment JSON files into a DataFrame."""
    rows = []
    for json_file in sorted(results_dir.glob("*_results*.json")):
        with json_file.open() as f:
            data = json.load(f)
        
        qaoa_data = data.get("qaoa", {})
        classical_data = data.get("classical", {})
        metrics = data.get("metrics", {})
        
        row = {
            "dataset": data.get("dataset", ""),
            "num_tasks": data.get("num_tasks"),
            "reps": data.get("config", {}).get("reps"),
            "qaoa_makespan": metrics.get("qaoa_makespan"),
            "classical_makespan": metrics.get("classical_makespan"),
            "relative_gap_pct": metrics.get("relative_gap_pct"),
            "qaoa_energy": qaoa_data.get("energy"),
            "qaoa_best_sample_energy": qaoa_data.get("best_sample_energy"),
            "qaoa_evaluations": qaoa_data.get("evaluations"),
            "balance_penalty": data.get("config", {}).get("balance_penalty_multiplier"),
            "priority_bias": data.get("config", {}).get("priority_bias"),
            "optimizer": data.get("config", {}).get("optimizer"),
            "restarts": data.get("config", {}).get("restarts"),
            "final_shots": data.get("config", {}).get("final_shots"),
            "file": str(json_file.name),
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def compute_statistics(df: pd.DataFrame, group_by: list[str] | None = None) -> pd.DataFrame:
    """Compute statistical summaries for grouped data."""
    if group_by is None:
        group_by = ["dataset", "reps"]
    
    agg_dict = {
        "qaoa_makespan": ["mean", "std", "min", "max", "count"],
        "classical_makespan": ["mean", "first"],  # Should be constant per dataset
        "relative_gap_pct": ["mean", "std", "min", "max", "count"],
        "qaoa_energy": ["mean", "std", "min", "count"],
        "qaoa_evaluations": ["mean"],
    }
    
    grouped = df.groupby(group_by).agg(agg_dict)
    
    # Flatten column names
    grouped.columns = ['_'.join(col).strip() if col[1] else col[0] 
                       for col in grouped.columns.values]
    
    # Compute confidence intervals (assuming normal distribution)
    # Use qaoa_makespan_count as reference count for all metrics (same N)
    reference_count_col = "qaoa_makespan_count"
    
    if reference_count_col not in grouped.columns:
        return grouped
    
    for col in ["qaoa_makespan", "relative_gap_pct"]:
        mean_col = f"{col}_mean"
        std_col = f"{col}_std"
        count_col = f"{col}_count"
        
        # Use reference count if specific count column doesn't exist
        use_count_col = count_col if count_col in grouped.columns else reference_count_col
        
        if mean_col in grouped.columns and std_col in grouped.columns and use_count_col in grouped.columns:
            # 95% confidence interval (t-distribution)
            counts = grouped[use_count_col].fillna(0).astype(int)
            valid_mask = counts > 1
            
            # Standard error of the mean
            sem = grouped[std_col] / np.sqrt(counts)
            sem[~valid_mask] = np.nan
            
            # Calculate t-critical for valid entries
            t_critical = pd.Series(np.nan, index=grouped.index, dtype=float)
            for idx in grouped.index[valid_mask]:
                count_val = int(counts.loc[idx])
                if count_val > 1:
                    try:
                        t_critical.loc[idx] = stats.t.ppf(0.975, count_val - 1)
                    except Exception:
                        pass
            
            # Compute confidence intervals
            grouped[f"{col}_ci_lower"] = grouped[mean_col] - t_critical * sem
            grouped[f"{col}_ci_upper"] = grouped[mean_col] + t_critical * sem
    
    return grouped


def compute_approximation_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Compute approximation ratio (QAOA makespan / optimal makespan)."""
    df = df.copy()
    df["approximation_ratio"] = df["qaoa_makespan"] / df["classical_makespan"]
    return df


def perform_statistical_tests(df: pd.DataFrame, group_by: str = "dataset") -> dict[str, Any]:
    """Perform statistical tests comparing QAOA across groups."""
    results = {}
    
    for dataset in df[group_by].unique():
        dataset_df = df[df[group_by] == dataset]
        
        if len(dataset_df) < 2:
            continue
        
        makespans = dataset_df["qaoa_makespan"].dropna()
        gaps = dataset_df["relative_gap_pct"].dropna()
        
        if len(makespans) < 2:
            continue
        
        # Normality test (Shapiro-Wilk for small samples)
        if len(makespans) <= 50:
            try:
                shapiro_stat, shapiro_p = stats.shapiro(makespans)
                results[f"{dataset}_normality"] = {
                    "statistic": float(shapiro_stat),
                    "p_value": float(shapiro_p),
                    "normal": shapiro_p > 0.05,
                }
            except Exception:
                pass
        
        # Descriptive statistics
        results[f"{dataset}_descriptive"] = {
            "mean": float(makespans.mean()),
            "median": float(makespans.median()),
            "std": float(makespans.std()),
            "min": float(makespans.min()),
            "max": float(makespans.max()),
            "q25": float(makespans.quantile(0.25)),
            "q75": float(makespans.quantile(0.75)),
            "n": int(len(makespans)),
        }
        
        # Confidence interval
        sem = makespans.std() / np.sqrt(len(makespans))
        t_critical = stats.t.ppf(0.975, len(makespans) - 1)
        ci_lower = makespans.mean() - t_critical * sem
        ci_upper = makespans.mean() + t_critical * sem
        results[f"{dataset}_ci"] = {
            "mean": float(makespans.mean()),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "confidence_level": 0.95,
        }
    
    return results


def print_summary_report(df: pd.DataFrame, output_path: Path | None = None) -> None:
    """Print a comprehensive statistical summary report."""
    lines = []
    lines.append("=" * 80)
    lines.append("STATISTICAL ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    # Overall statistics
    lines.append("OVERALL STATISTICS")
    lines.append("-" * 80)
    lines.append(f"Total experiments: {len(df)}")
    lines.append(f"Unique datasets: {df['dataset'].nunique()}")
    lines.append(f"Average QAOA makespan: {df['qaoa_makespan'].mean():.2f} ± {df['qaoa_makespan'].std():.2f}")
    lines.append(f"Average relative gap: {df['relative_gap_pct'].mean():.2f}% ± {df['relative_gap_pct'].std():.2f}%")
    lines.append("")
    
    # Per-dataset statistics
    lines.append("PER-DATASET STATISTICS")
    lines.append("-" * 80)
    grouped_stats = compute_statistics(df, group_by=["dataset", "reps"])
    
    for idx, (key, row) in enumerate(grouped_stats.iterrows()):
        dataset, reps = key
        lines.append(f"\n{dataset} (p={reps}):")
        lines.append(f"  QAOA makespan: {row.get('qaoa_makespan_mean', 0):.2f} ± {row.get('qaoa_makespan_std', 0):.2f}")
        if 'qaoa_makespan_ci_lower' in row:
            lines.append(f"    95% CI: [{row['qaoa_makespan_ci_lower']:.2f}, {row['qaoa_makespan_ci_upper']:.2f}]")
        lines.append(f"  Relative gap: {row.get('relative_gap_pct_mean', 0):.2f}% ± {row.get('relative_gap_pct_std', 0):.2f}%")
        lines.append(f"  N = {int(row.get('qaoa_makespan_count', 0))}")
    
    lines.append("")
    
    # Approximation ratios
    lines.append("APPROXIMATION RATIOS")
    lines.append("-" * 80)
    df_with_ratios = compute_approximation_ratios(df)
    for dataset in df['dataset'].unique():
        dataset_data = df_with_ratios[df_with_ratios['dataset'] == dataset]
        ratios = dataset_data['approximation_ratio'].dropna()
        if len(ratios) > 0:
            lines.append(f"{dataset}: {ratios.mean():.3f} ± {ratios.std():.3f} (range: [{ratios.min():.3f}, {ratios.max():.3f}])")
    lines.append("")
    
    # Statistical tests
    lines.append("STATISTICAL TESTS")
    lines.append("-" * 80)
    test_results = perform_statistical_tests(df)
    for key, value in test_results.items():
        if "descriptive" in key:
            lines.append(f"\n{key}:")
            for stat_name, stat_value in value.items():
                lines.append(f"  {stat_name}: {stat_value}")
        elif "ci" in key:
            lines.append(f"\n{key} (95% Confidence Interval):")
            lines.append(f"  Mean: {value['mean']:.2f}")
            lines.append(f"  CI: [{value['ci_lower']:.2f}, {value['ci_upper']:.2f}]")
    
    lines.append("")
    lines.append("=" * 80)
    
    report_text = "\n".join(lines)
    print(report_text)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_text)
        print(f"\nReport saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Statistical analysis of experiment results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent.parent / "results",
        help="Directory containing experiment JSON files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for statistical report",
    )
    args = parser.parse_args()
    
    if not args.results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {args.results_dir}")
    
    df = load_experiment_results(args.results_dir)
    
    if df.empty:
        print("No experiment results found.")
        return
    
    output_path = args.output or (args.results_dir.parent / "analysis" / "statistical_report.txt")
    print_summary_report(df, output_path)
    
    # Also save CSV summary
    csv_path = args.results_dir.parent / "analysis" / "statistical_summary.csv"
    stats_df = compute_statistics(df, group_by=["dataset", "reps"])
    stats_df.to_csv(csv_path)
    print(f"\nStatistical summary CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()

