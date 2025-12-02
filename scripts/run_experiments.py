"""Convenience CLI for running the QAOA and classical baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from quantum_scheduler import solve_qaoa_local
from quantum_scheduler.classical_solver import solve_classical
from quantum_scheduler.utils import load_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="dataset_5.csv",
        help="Dataset filename or absolute path (default: dataset_5.csv)",
    )
    parser.add_argument("--reps", type=int, default=1, help="Number of QAOA layers")
    parser.add_argument(
        "--maxiter",
        type=int,
        default=20,
        help="Maximum optimizer iterations",
    )
    parser.add_argument(
        "--machines",
        type=int,
        default=3,
        help="Number of identical machines for the classical baseline",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON path (defaults to results/<dataset>.json)",
    )
    return parser.parse_args()


def resolve_dataset_path(dataset_arg: str) -> Path:
    dataset_path = Path(dataset_arg)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / "data" / "datasets" / dataset_arg
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    return dataset_path


def main() -> None:
    args = parse_args()
    dataset_path = resolve_dataset_path(args.dataset)
    tasks = load_tasks(dataset_path)

    qaoa_res = solve_qaoa_local(tasks, reps=args.reps, maxiter=args.maxiter)
    classical_res = solve_classical([task["p_i"] for task in tasks], M=args.machines)

    payload = {
        "dataset": dataset_path.name,
        "num_tasks": len(tasks),
        "qaoa": qaoa_res,
        "classical": classical_res,
    }

    output_path = Path(args.output) if args.output else ROOT / "results" / f"{dataset_path.stem}_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
