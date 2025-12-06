"""Plot helper for experiment outputs produced by run_experiments.py."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from quantum_scheduler.utils import load_tasks, qubo_from_tasks  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default=ROOT / "results",
        type=Path,
        help="Directory that contains *_results.json files",
    )
    parser.add_argument(
        "--plots-dir",
        default=ROOT / "results" / "plots",
        type=Path,
        help="Directory where plots will be saved",
    )
    return parser.parse_args()


def load_result_files(results_dir: Path) -> list[dict]:
    files = sorted(results_dir.glob("*_results.json"))
    if not files:
        raise FileNotFoundError(
            f"No results found under {results_dir}. Run scripts/run_experiments.py first."
        )

    payloads = []
    for path in files:
        payloads.append(json.loads(path.read_text()))
    payloads.sort(key=lambda entry: entry.get("num_tasks", 0))
    return payloads


def _dataset_stem(entry: dict) -> str:
    dataset = entry.get("dataset") or Path(entry.get("dataset_path", "dataset")).name
    return Path(dataset).stem


def _resolve_dataset_path(entry: dict) -> Path | None:
    path = entry.get("dataset_path")
    if path:
        resolved = Path(path)
        if resolved.exists():
            return resolved
    fallback = ROOT / "data" / "datasets" / entry.get("dataset", "")
    if fallback.exists():
        return fallback
    print(f"[plot_results] Skipping dataset load for {entry.get('dataset')} (file missing)")
    return None


def _load_tasks(entry: dict) -> list[dict] | None:
    dataset_path = _resolve_dataset_path(entry)
    if dataset_path is None:
        return None
    return load_tasks(dataset_path)


def _compute_qubo_matrix(entry: dict) -> np.ndarray | None:
    tasks = _load_tasks(entry)
    if not tasks:
        return None
    qaoa_cfg = entry.get("qaoa", {})
    config = entry.get("config", {})
    penalty = qaoa_cfg.get("balance_penalty_multiplier") or config.get("balance_penalty_multiplier")
    priority = qaoa_cfg.get("priority_bias") or config.get("priority_bias", 0.1)
    Q, *_ = qubo_from_tasks(
        tasks,
        balance_penalty_multiplier=penalty,
        priority_bias=priority,
    )
    return Q


def plot_classical(data: list[dict], plots_dir: Path) -> Path | None:
    if not data:
        return None
    sizes = [entry["num_tasks"] for entry in data]
    makespans = [entry["classical"]["makespan"] for entry in data]

    plt.figure(figsize=(8, 5))
    plt.plot(sizes, makespans, marker="o", label="Classical optimum")
    plt.title("Classical makespan vs dataset size")
    plt.xlabel("# jobs")
    plt.ylabel("Makespan")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path = plots_dir / "classical_makespans.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")
    return output_path


def plot_qaoa_energy(data: list[dict], plots_dir: Path) -> Path | None:
    if not data:
        return None
    sizes = [entry["num_tasks"] for entry in data]
    energies = [entry["qaoa"]["energy"] for entry in data]

    plt.figure(figsize=(8, 5))
    plt.plot(sizes, energies, marker="o", color="purple", label="QAOA energy")
    plt.title("QAOA energy vs dataset size")
    plt.xlabel("# jobs")
    plt.ylabel("Energy (expectation value)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path = plots_dir / "qaoa_energy.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")
    return output_path


def plot_makespan_comparison(data: list[dict], plots_dir: Path) -> Path | None:
    entries = []
    for entry in data:
        metrics = entry.get("metrics") or {}
        if metrics.get("qaoa_makespan") is None or metrics.get("classical_makespan") is None:
            continue
        entries.append(
            (
                entry.get("dataset", f"dataset_{entry.get('num_tasks')}"),
                metrics["qaoa_makespan"],
                metrics["classical_makespan"],
            )
        )
    if not entries:
        print("[plot_results] No makespan metrics available; skipping bar chart.")
        return None

    labels, qaoa_vals, classical_vals = zip(*entries)
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, classical_vals, width, label="Classical", color="#4c72b0")
    plt.bar(x + width / 2, qaoa_vals, width, label="QAOA", color="#dd8452")
    plt.title("Makespan comparison")
    plt.ylabel("Makespan")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    output_path = plots_dir / "makespan_comparison.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")
    return output_path


def plot_convergence_curves(data: list[dict], plots_dir: Path) -> Path | None:
    plotted = False
    plt.figure(figsize=(10, 6))
    for entry in data:
        trace = entry.get("qaoa", {}).get("energy_trace") or []
        if not trace:
            continue
        plotted = True
        energies = [point.get("energy") for point in trace]
        xs = list(range(1, len(energies) + 1))
        label = f"{_dataset_stem(entry)} ({entry.get('num_tasks', '?')} jobs)"
        plt.plot(xs, energies, label=label, linewidth=1.2)

    if not plotted:
        plt.close()
        print("[plot_results] No convergence traces found; skipping convergence curve.")
        return None

    plt.title("QAOA optimizer convergence")
    plt.xlabel("Energy evaluation #")
    plt.ylabel("Energy value")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path = plots_dir / "convergence_curves.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")
    return output_path


def _binary_vector(value: int, num_bits: int) -> np.ndarray:
    bits = [(value >> bit) & 1 for bit in range(num_bits)]
    return np.array(bits[::-1], dtype=float)


def _sample_qubo_energies(Q: np.ndarray, max_samples: int = 4096) -> list[float]:
    n = Q.shape[0]
    total_states = 1 << n
    if total_states <= max_samples:
        indices: Iterable[int] = range(total_states)
    else:
        rng = np.random.default_rng(0)
        indices = rng.choice(total_states, size=max_samples, replace=False)

    energies: list[float] = []
    for idx in indices:
        vec = _binary_vector(int(idx), n)
        energy = float(vec @ Q @ vec)
        energies.append(energy)
    return energies


def plot_qubo_landscape(entry: dict, plots_dir: Path) -> Path | None:
    Q = _compute_qubo_matrix(entry)
    if Q is None:
        return None
    energies = _sample_qubo_energies(Q)
    if not energies:
        return None
    sorted_energies = np.sort(energies)
    plt.figure(figsize=(8, 5))
    plt.plot(sorted_energies, color="#55a868")
    plt.title(f"QUBO landscape ({_dataset_stem(entry)})")
    plt.xlabel("Configuration index (sorted)")
    plt.ylabel("Energy")
    plt.tight_layout()
    output_path = plots_dir / f"{_dataset_stem(entry)}_qubo_landscape.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")
    return output_path


def plot_qubo_heatmap(entry: dict, plots_dir: Path) -> Path | None:
    Q = _compute_qubo_matrix(entry)
    if Q is None:
        return None
    plt.figure(figsize=(6, 5))
    im = plt.imshow(Q, cmap="coolwarm", aspect="auto")
    plt.title(f"QUBO coefficients heatmap ({_dataset_stem(entry)})")
    plt.xlabel("Variable index")
    plt.ylabel("Variable index")
    plt.colorbar(im, label="Q_ij")
    plt.tight_layout()
    output_path = plots_dir / f"{_dataset_stem(entry)}_qubo_heatmap.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")
    return output_path


def plot_sample_histogram(entry: dict, plots_dir: Path, top_k: int = 12) -> Path | None:
    counts = entry.get("qaoa", {}).get("counts") or {}
    if not counts:
        print(f"[plot_results] No sample counts for {_dataset_stem(entry)}; skipping histogram.")
        return None

    total = sum(counts.values())
    if total == 0:
        return None

    top_items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    labels = [bit for bit, _ in top_items]
    probs = [count / total for _, count in top_items]
    x = np.arange(len(labels))

    plt.figure(figsize=(max(8, len(labels)), 4))
    plt.bar(x, probs, color="#8172b2")
    plt.title(f"Top-{len(labels)} sample distribution ({_dataset_stem(entry)})")
    plt.ylabel("Probability")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.tight_layout()
    output_path = plots_dir / f"{_dataset_stem(entry)}_sample_histogram.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    data = load_result_files(args.results_dir)
    plot_classical(data, args.plots_dir)
    plot_qaoa_energy(data, args.plots_dir)
    plot_makespan_comparison(data, args.plots_dir)
    plot_convergence_curves(data, args.plots_dir)

    for entry in data:
        plot_qubo_landscape(entry, args.plots_dir)
        plot_qubo_heatmap(entry, args.plots_dir)
        plot_sample_histogram(entry, args.plots_dir)


if __name__ == "__main__":
    main()
