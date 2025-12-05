import json
from collections import Counter
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.dataset_loader import load_tasks

RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
DATASETS_DIR = Path("datasets")


def ensure_directories() -> None:
    """Guarantee that the output folders exist."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    """Load helper with friendlier error messaging."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required file '{path}' is missing. "
            "Generate results first with your experiment runner."
        )
    with path.open() as f:
        return json.load(f)


def infer_sizes(qaoa_data, classical_data):
    sizes = [
        entry.get("dataset_size") or entry.get("size")
        for entry in qaoa_data
        if entry.get("dataset_size") or entry.get("size")
    ]
    if not sizes:
        sizes = [
            entry.get("dataset_size") or entry.get("size")
            for entry in classical_data
            if entry.get("dataset_size") or entry.get("size")
        ]
    if sizes:
        return sizes
    # Fallback to positional sizes if metadata missing
    total = max(len(qaoa_data), len(classical_data))
    return list(range(1, total + 1))


def compute_relative_gap(qaoa, classical):
    return [
        100 * (q - c) / c if c else 0.0
        for q, c in zip(qaoa, classical)
    ]


def plot_makespan_bar(sizes, qaoa, classical):
    indices = np.arange(len(sizes))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(indices - width / 2, qaoa, width, label="QAOA")
    plt.bar(indices + width / 2, classical, width, label="Classical Optimum")
    plt.xticks(indices, sizes)
    plt.xlabel("Dataset size (# jobs)")
    plt.ylabel("Makespan")
    plt.title("Makespan Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "makespan_comparison_bar.png")
    plt.close()


def plot_relative_gap(sizes, gap):
    plt.figure(figsize=(9, 5))
    plt.plot(sizes, gap, marker="o", color="crimson")
    plt.xlabel("Dataset size (# jobs)")
    plt.ylabel("Gap %")
    plt.title("Relative Performance Gap (%)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "relative_gap.png")
    plt.close()


def build_qubo_matrix_for_size(size):
    dataset_file = DATASETS_DIR / f"dataset_{size}.csv"
    if not dataset_file.exists():
        print(f"[warn] Dataset file {dataset_file} not found. Skipping QUBO visuals.")
        return None

    tasks = load_tasks(str(dataset_file))
    diag = []
    for task in tasks:
        p = float(task.get("p_i", task.get("p", 0.0)))
        w = float(task.get("priority_w", task.get("w", 1.0)))
        diag.append(p * w)
    return np.diag(diag)


def plot_qubo_landscape(Q, size):
    n = Q.shape[0]
    energies = []
    for bit_tuple in product([0, 1], repeat=n):
        x = np.array(bit_tuple)
        energies.append(float(x @ Q @ x))

    sorted_energies = np.sort(energies)
    plt.figure(figsize=(10, 5))
    plt.plot(sorted_energies, linewidth=1.2)
    plt.title(f"QUBO Energy Landscape (dataset {size})")
    plt.xlabel("Configuration index (sorted by energy)")
    plt.ylabel("Energy")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"qubo_landscape_size_{size}.png")
    plt.close()


def plot_qubo_heatmap(Q, size):
    plt.figure(figsize=(6, 5))
    im = plt.imshow(Q, cmap="viridis")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Coefficient value")
    plt.title(f"QUBO Coefficient Heatmap (dataset {size})")
    plt.xlabel("Variable index")
    plt.ylabel("Variable index")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"qubo_heatmap_size_{size}.png")
    plt.close()


def plot_convergence_curves(qaoa_data):
    traces = []
    for entry in qaoa_data:
        trace = (
            entry.get("energy_trace")
            or entry.get("objective_values")
            or entry.get("convergence")
        )
        if trace:
            traces.append((entry.get("dataset_size", "?"), trace))

    if not traces:
        print("[warn] No convergence trace metadata found. Skipping convergence plot.")
        return

    plt.figure(figsize=(9, 5))
    for size, trace in traces:
        plt.plot(range(len(trace)), trace, marker="o", label=f"n={size}")

    plt.xlabel("Iteration")
    plt.ylabel("Objective (energy)")
    plt.title("QAOA Convergence")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "qaoa_convergence.png")
    plt.close()


def plot_sample_distribution(qaoa_data, top_k=15):
    for entry in qaoa_data:
        raw = (
            entry.get("sample_counts")
            or entry.get("bitstring_counts")
            or entry.get("samples")
        )
        if isinstance(raw, dict) and raw:
            counts = Counter(raw)
        elif isinstance(raw, list) and raw:
            counts = Counter(raw)
        else:
            counts = None

        if counts:
            dataset_size = entry.get("dataset_size", "unknown")
            break
    else:
        print("[warn] No bitstring frequency data found. Skipping sample histogram.")
        return

    most_common = counts.most_common(top_k)
    labels = [item[0] for item in most_common]
    values = [item[1] for item in most_common]

    plt.figure(figsize=(max(8, 0.5 * len(labels)), 5))
    plt.bar(range(len(labels)), values)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    plt.xlabel("Bitstring")
    plt.ylabel("Counts")
    plt.title(f"Top-{top_k} Sample Distribution (dataset {dataset_size})")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"sample_histogram_size_{dataset_size}.png")
    plt.close()


def main():
    ensure_directories()
    qaoa_data = load_json(RESULTS_DIR / "qaoa_results.json")
    classical_data = load_json(RESULTS_DIR / "classical_results.json")

    qaoa_makespans = [entry["makespan"] for entry in qaoa_data]
    classical_makespans = [entry["makespan"] for entry in classical_data]
    sizes = infer_sizes(qaoa_data, classical_data)
    gap = compute_relative_gap(qaoa_makespans, classical_makespans)

    plot_makespan_bar(sizes, qaoa_makespans, classical_makespans)
    plot_relative_gap(sizes, gap)

    plot_convergence_curves(qaoa_data)
    plot_sample_distribution(qaoa_data)

    if sizes:
        qubo_size = sizes[-1]
        Q = build_qubo_matrix_for_size(qubo_size)
        if Q is not None:
            plot_qubo_landscape(Q, qubo_size)
            plot_qubo_heatmap(Q, qubo_size)

    print(f"Plots saved in {PLOTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
