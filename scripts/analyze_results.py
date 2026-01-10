"""Aggregate experiment outputs and generate publication-ready tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd
from tabulate import tabulate

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results", help="Directory with *_results.json files")
    parser.add_argument("--input-csv", type=Path, default=None, help="Optional CSV summary (e.g., produced by run_experiments --config)")
    parser.add_argument("--output-md", type=Path, default=ROOT / "analysis" / "summary.md", help="Path for the markdown report")
    parser.add_argument("--output-tex", type=Path, default=ROOT / "analysis" / "summary.tex", help="Path for the LaTeX table")
    return parser.parse_args()


def load_json_results(results_dir: Path) -> pd.DataFrame:
    rows: List[dict] = []
    for path in sorted(results_dir.glob("*_results*.json")):
        data = json.loads(path.read_text())
        metrics = data.get("metrics", {})
        rows.append(
            {
                "dataset": data.get("dataset"),
                "num_tasks": data.get("num_tasks"),
                "reps": data.get("config", {}).get("reps"),
                "shots": data.get("config", {}).get("final_shots"),
                "qaoa_makespan": metrics.get("qaoa_makespan"),
                "min_makespan_in_samples": metrics.get("min_makespan_in_samples"),
                "classical_makespan": metrics.get("classical_makespan"),
                "relative_gap_pct": metrics.get("relative_gap_pct"),
                "best_sample_gap_pct": metrics.get("best_sample_gap_pct"),
                "classical_method": data.get("classical_method", "brute_force"),
                "makespan_penalty": data.get("config", {}).get("makespan_penalty_multiplier"),
                "path": str(path),
            }
        )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> str:
    grouped = df.groupby("dataset").agg(
        qaoa_mean=("relative_gap_pct", "mean"),
        qaoa_std=("relative_gap_pct", "std"),
        qaoa_best=("relative_gap_pct", "min"),
    )
    grouped = grouped.round(3)
    return tabulate(grouped, headers="keys", tablefmt="github")


def save_outputs(df: pd.DataFrame, args: argparse.Namespace) -> None:
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)

    md_table = tabulate(df, headers="keys", tablefmt="github", showindex=False)
    args.output_md.write_text(f"# Experimental Summary\n\n{md_table}\n")

    tex_table = tabulate(df, headers="keys", tablefmt="latex", showindex=False)
    args.output_tex.write_text(tex_table)

    print(f"Markdown summary written to {args.output_md}")
    print(f"LaTeX table written to {args.output_tex}")


def main() -> None:
    args = parse_args()

    if args.input_csv and args.input_csv.exists():
        df = pd.read_csv(args.input_csv)
    else:
        df = load_json_results(args.results_dir)

    if df.empty:
        raise SystemExit("No experiment results found to analyse.")

    print("\nRelative performance summary by dataset:")
    print(summarize(df))

    save_outputs(df, args)


if __name__ == "__main__":
    main()
