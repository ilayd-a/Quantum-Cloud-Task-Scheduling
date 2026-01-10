"""Analyze correlation between QUBO energy and makespan across different problem sizes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from quantum_scheduler.utils import load_tasks
from quantum_scheduler.utils.qubo_builder import qubo_from_tasks
from quantum_scheduler.utils.decoder import bitstring_to_vector, decode_solution_vector
from quantum_scheduler.qaoa_solver import qubo_to_ising


def compute_correlation_for_dataset(
    dataset_path: Path,
    balance_penalty: float = 10.0,
    makespan_penalty: float = 30.0,
    priority_bias: float = 0.1,
    num_samples: int = 2000,
) -> dict:
    """Compute correlation metrics for a single dataset."""
    tasks = load_tasks(dataset_path)
    processing_times = [float(task["p_i"]) for task in tasks]
    n = len(tasks)
    
    # Build QUBO
    Q, _, _, _ = qubo_from_tasks(
        tasks,
        balance_penalty_multiplier=balance_penalty,
        priority_bias=priority_bias,
        makespan_penalty_multiplier=makespan_penalty,
    )
    
    # Sample random bitstrings
    rng = np.random.default_rng(42)
    samples = []
    
    for _ in range(num_samples):
        bits = rng.integers(0, 2, size=n)
        bitstring = "".join(str(b) for b in bits)
        
        # Compute QUBO energy
        vec = bitstring_to_vector(bitstring, n)
        qubo_energy = float(vec @ Q @ vec)
        
        # Decode to makespan
        schedule = decode_solution_vector(vec, processing_times)
        makespan = schedule["makespan"]
        loads = schedule["loads"]
        load_imbalance = abs(loads[0] - loads[1])
        
        samples.append({
            "qubo_energy": qubo_energy,
            "makespan": makespan,
            "load_imbalance": load_imbalance,
        })
    
    df = pd.DataFrame(samples)
    
    # Compute correlations
    pearson_qm = df["qubo_energy"].corr(df["makespan"])
    spearman_qm, spearman_p = spearmanr(df["qubo_energy"], df["makespan"])
    
    # Compute top-k overlap
    k = min(100, len(df) // 10)  # Top 10% or 100, whichever smaller
    top_energy = df.nsmallest(k, "qubo_energy")
    top_makespan = df.nsmallest(k, "makespan")
    overlap = len(set(top_energy.index) & set(top_makespan.index)) / k
    
    return {
        "dataset": dataset_path.name,
        "num_tasks": n,
        "num_samples": num_samples,
        "pearson_correlation": float(pearson_qm),
        "spearman_correlation": float(spearman_qm),
        "spearman_p_value": float(spearman_p),
        "top_k_overlap": float(overlap),
        "k": k,
        "energy_mean": float(df["qubo_energy"].mean()),
        "energy_std": float(df["qubo_energy"].std()),
        "makespan_mean": float(df["makespan"].mean()),
        "makespan_std": float(df["makespan"].std()),
        "makespan_min": float(df["makespan"].min()),
        "makespan_max": float(df["makespan"].max()),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze QUBO-makespan correlation scaling")
    parser.add_argument("--datasets", type=Path, nargs="+", help="Dataset CSV files to analyze")
    parser.add_argument("--datasets-dir", type=Path, default=ROOT / "data" / "datasets", help="Directory with datasets")
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 20, 25], help="Problem sizes to analyze")
    parser.add_argument("--balance-penalty", type=float, default=10.0, help="Balance penalty")
    parser.add_argument("--makespan-penalty", type=float, default=30.0, help="Makespan penalty")
    parser.add_argument("--priority-bias", type=float, default=0.1, help="Priority bias")
    parser.add_argument("--num-samples", type=int, default=2000, help="Number of random samples")
    parser.add_argument("--output", type=Path, help="Output JSON file")
    parser.add_argument("--output-csv", type=Path, help="Output CSV file")
    
    args = parser.parse_args()
    
    if args.datasets:
        dataset_paths = args.datasets
    else:
        dataset_paths = [args.datasets_dir / f"dataset_{size}.csv" for size in args.sizes]
        dataset_paths = [p for p in dataset_paths if p.exists()]
    
    if not dataset_paths:
        print("Error: No datasets found")
        return
    
    results = []
    for dataset_path in dataset_paths:
        print(f"Analyzing {dataset_path.name}...")
        try:
            result = compute_correlation_for_dataset(
                dataset_path,
                balance_penalty=args.balance_penalty,
                makespan_penalty=args.makespan_penalty,
                priority_bias=args.priority_bias,
                num_samples=args.num_samples,
            )
            results.append(result)
            print(f"  Pearson: {result['pearson_correlation']:.4f}, Spearman: {result['spearman_correlation']:.4f}, Overlap: {result['top_k_overlap']:.2%}")
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("CORRELATION SCALING ANALYSIS")
    print("=" * 80)
    print(df.to_string(index=False))
    
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"CSV saved to: {args.output_csv}")


if __name__ == "__main__":
    main()

