"""Advanced plotting utilities for QAOA scheduling experiments.

This module loads JSON experiment outputs produced by ``run_experiments.py``
and generates publication-ready plots in the style commonly requested by
IEEE/QCE reviewers. The figures cover convergence behaviour, depth scaling,
classical vs quantum comparisons, sample distributions, correlation analyses,
and exhaustive (small-n) energy landscapes.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from quantum_scheduler.utils.decoder import bitstring_to_vector, decode_solution_vector
from quantum_scheduler.utils.qubo_builder import qubo_from_tasks

sns.set_theme(style="whitegrid", palette="colorblind")

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DatasetArtifacts:
    tasks: list[dict]
    qubo: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results",
        help="Directory containing *_results.json payloads",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=ROOT / "results" / "plots",
        help="Directory where figures will be stored",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=ROOT / "data" / "datasets",
        help="Directory containing source CSV datasets",
    )
    parser.add_argument(
        "--max-landscape-qubits",
        type=int,
        default=12,
        help="Maximum problem size (n) for exhaustive landscape plots",
    )
    return parser.parse_args()


def load_results(results_dir: Path) -> list[dict]:
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    payloads: list[dict] = []
    for path in sorted(results_dir.glob("*_results.json")):
        try:
            payloads.append(json.loads(path.read_text()))
        except json.JSONDecodeError as exc:
            print(f"[advanced_plots] Skipping {path.name}: {exc}")
    if not payloads:
        raise FileNotFoundError(f"No *_results.json files found under {results_dir}")
    return payloads


def _dataset_stem(entry: dict) -> str:
    dataset = entry.get("dataset") or Path(entry.get("dataset_path", "dataset")).name
    return Path(dataset).stem


def _resolve_dataset_path(entry: dict, datasets_dir: Path) -> Path | None:
    explicit = entry.get("dataset_path")
    if explicit and Path(explicit).exists():
        return Path(explicit)
    candidate = datasets_dir / entry.get("dataset", "")
    if candidate.exists():
        return candidate
    print(f"[advanced_plots] Missing dataset for {_dataset_stem(entry)}")
    return None


def _load_tasks(entry: dict, datasets_dir: Path) -> list[dict] | None:
    dataset_path = _resolve_dataset_path(entry, datasets_dir)
    if dataset_path is None:
        return None
    rows: list[dict] = []
    with dataset_path.open() as fh:
        header = fh.readline().strip().split(",")
        for idx, line in enumerate(fh, start=1):
            fields = dict(zip(header, line.strip().split(",")))
            duration = fields.get("p_i") or fields.get("p") or fields.get("duration")
            if duration is None:
                continue
            try:
                p_i = float(duration)
            except ValueError:
                continue
            priority_raw = fields.get("priority_w") or fields.get("weight")
            try:
                priority = float(priority_raw) if priority_raw else 1.0
            except ValueError:
                priority = 1.0
            rows.append(
                {
                    "job": fields.get("job") or fields.get("task") or idx,
                    "p_i": p_i,
                    "priority_w": priority,
                }
            )
    return rows or None


def _ensure_artifacts(
    entry: dict, datasets_dir: Path, cache: dict[str, DatasetArtifacts]
) -> DatasetArtifacts | None:
    stem = _dataset_stem(entry)
    if stem in cache:
        return cache[stem]
    tasks = _load_tasks(entry, datasets_dir)
    if not tasks:
        return None
    penalty = entry.get("qaoa", {}).get("balance_penalty_multiplier") or entry.get("config", {}).get(
        "balance_penalty_multiplier", 10.0
    )
    priority = entry.get("qaoa", {}).get("priority_bias") or entry.get("config", {}).get("priority_bias", 0.1)
    qubo, *_ = qubo_from_tasks(tasks, balance_penalty_multiplier=penalty, priority_bias=priority)
    artifacts = DatasetArtifacts(tasks=tasks, qubo=qubo)
    cache[stem] = artifacts
    return artifacts


def _bitstring_energy(bitstring: str, Q: np.ndarray) -> float:
    vec = bitstring_to_vector(bitstring, Q.shape[0])
    return float(vec @ Q @ vec)


def _bitstring_makespan(bitstring: str, tasks: Iterable[dict]) -> float:
    durations = [float(task["p_i"]) for task in tasks]
    vec = bitstring_to_vector(bitstring, len(durations))
    schedule = decode_solution_vector(vec, durations)
    return float(schedule["makespan"])


def plot_optimizer_convergence(grouped: dict[str, list[dict]], plots_dir: Path) -> None:
    for dataset, entries in grouped.items():
        plt.figure(figsize=(9, 5.5))
        plotted = False
        for entry in entries:
            trace = (entry.get("qaoa") or {}).get("energy_trace") or entry.get("energy_trace")
            if not trace:
                continue
            restarts = defaultdict(list)
            for point in trace:
                if point.get("energy") is None:
                    continue
                restart_id = int(point.get("restart") or 1)
                restarts[restart_id].append(point)
            optimizer = (entry.get("config") or {}).get("optimizer", "unknown").upper()
            for restart_id, points in restarts.items():
                points.sort(key=lambda p: p.get("evaluation") or p.get("iteration") or p.get("step") or 0)
                energies = np.array([float(p["energy"]) for p in points], dtype=float)
                if energies.size == 0:
                    continue
                running_best = np.minimum.accumulate(energies)
                evals = [
                    int(p.get("evaluation") or p.get("iteration") or p.get("step") or idx + 1)
                    for idx, p in enumerate(points)
                ]
                label = f"{optimizer} restart {restart_id} (shots={entry.get('config', {}).get('shots', '?')})"
                plt.plot(evals[: running_best.size], running_best, linewidth=1.4, label=label)
                plotted = True
        if not plotted:
            plt.close()
            print(f"[advanced_plots] Skipping convergence plot for {dataset} (no trace).")
            continue
        plt.title(f"QAOA optimizer convergence — {dataset}")
        plt.xlabel("Function evaluation")
        plt.ylabel("Best energy observed")
        plt.legend(fontsize=8)
        plt.tight_layout()
        output = plots_dir / f"convergence_{dataset}.png"
        plt.savefig(output)
        plt.close()
        print(f"Saved {output}")


def plot_depth_scaling(grouped: dict[str, list[dict]], plots_dir: Path) -> None:
    rows = []
    for dataset, entries in grouped.items():
        for entry in entries:
            config = entry.get("config") or {}
            qaoa = entry.get("qaoa") or {}
            metrics = entry.get("metrics") or {}
            reps = config.get("reps")
            makespan = entry.get("qaoa_makespan") or metrics.get("qaoa_makespan")
            energy = qaoa.get("energy") or metrics.get("qaoa_energy")
            if reps is None or makespan is None or energy is None:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "reps": int(reps),
                    "makespan": float(makespan),
                    "energy": float(energy),
                }
            )
    if not rows:
        print("[advanced_plots] No data for depth scaling plot.")
        return

    plt.figure(figsize=(9, 5.5))
    reps = [row["reps"] for row in rows]
    makespans = [row["makespan"] for row in rows]
    energies = [row["energy"] for row in rows]
    sns.lineplot(x=reps, y=makespans, marker="o", label="Final makespan", color="#1f77b4")
    ax = plt.gca()
    ax.set_xlabel("QAOA depth (p)")
    ax.set_ylabel("Final makespan")

    ax_energy = ax.twinx()
    sns.lineplot(x=reps, y=energies, marker="s", label="Final energy", color="#ff7f0e", ax=ax_energy)
    ax_energy.set_ylabel("Final energy")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax_energy.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")
    plt.title("Depth scaling across datasets")
    plt.tight_layout()
    output = plots_dir / "depth_scaling.png"
    plt.savefig(output)
    plt.close()
    print(f"Saved {output}")


def plot_classical_vs_qaoa(grouped: dict[str, list[dict]], plots_dir: Path) -> None:
    datasets = []
    classical_vals = []
    qaoa_vals = []
    for dataset, entries in grouped.items():
        classical = []
        qaoa = []
        for entry in entries:
            metrics = entry.get("metrics") or {}
            classical_ms = metrics.get("classical_makespan") or (entry.get("classical") or {}).get("makespan")
            qaoa_ms = entry.get("qaoa_makespan") or metrics.get("qaoa_makespan")
            if classical_ms is None or qaoa_ms is None:
                continue
            classical.append(float(classical_ms))
            qaoa.append(float(qaoa_ms))
        if not classical or not qaoa:
            continue
        datasets.append(dataset)
        classical_vals.append(np.mean(classical))
        qaoa_vals.append(np.mean(qaoa))

    if not datasets:
        print("[advanced_plots] No data for classical vs QAOA plot.")
        return

    x = np.arange(len(datasets))
    width = 0.35
    plt.figure(figsize=(max(10, len(datasets) * 1.2), 5.5))
    plt.bar(x - width / 2, classical_vals, width, label="Classical", color="#4c72b0")
    plt.bar(x + width / 2, qaoa_vals, width, label="QAOA", color="#dd8452")

    for idx, (c_val, q_val) in enumerate(zip(classical_vals, qaoa_vals)):
        if c_val == 0:
            continue
        gap = 100.0 * (q_val - c_val) / c_val
        plt.text(
            x[idx],
            max(c_val, q_val) * 1.01,
            f"{gap:+.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
        )

    plt.xticks(x, datasets, rotation=20, ha="right")
    plt.ylabel("Makespan")
    plt.title("Classical vs QAOA makespan comparison")
    plt.legend()
    plt.tight_layout()
    output = plots_dir / "classical_vs_qaoa.png"
    plt.savefig(output)
    plt.close()
    print(f"Saved {output}")


def plot_bitstring_histograms(
    grouped: dict[str, list[dict]], plots_dir: Path, datasets_dir: Path, cache: dict[str, DatasetArtifacts]
) -> None:
    for dataset, entries in grouped.items():
        aggregate = Counter()
        artifacts = None
        for entry in entries:
            artifacts = artifacts or _ensure_artifacts(entry, datasets_dir, cache)
            counts = (entry.get("qaoa") or {}).get("counts") or {}
            aggregate.update({bit.replace(" ", ""): int(freq) for bit, freq in counts.items()})
        if not aggregate or artifacts is None:
            print(f"[advanced_plots] Skipping histogram for {dataset} (missing counts).")
            continue

        total = sum(aggregate.values())
        top_items = aggregate.most_common(15)
        bits = [bit for bit, _ in top_items]
        probs = [freq / total for _, freq in top_items]
        energies = [_bitstring_energy(bit, artifacts.qubo) for bit in bits]

        indices = np.arange(len(bits))
        fig, ax_prob = plt.subplots(figsize=(max(10, len(bits) * 0.6), 5))
        ax_prob.bar(indices, probs, color="#8172b2", alpha=0.85, label="Probability")
        ax_prob.set_ylabel("Probability")
        ax_prob.set_xticks(indices)
        ax_prob.set_xticklabels(bits, rotation=35, ha="right")

        ax_energy = ax_prob.twinx()
        ax_energy.plot(indices, energies, color="#c44e52", marker="s", linewidth=1.2, label="Energy")
        ax_energy.set_ylabel("Energy")

        handles, labels = ax_prob.get_legend_handles_labels()
        handles2, labels2 = ax_energy.get_legend_handles_labels()
        ax_prob.legend(handles + handles2, labels + labels2, loc="upper right")
        ax_prob.set_title(f"Top-{len(bits)} bitstring distribution — {dataset}")
        fig.tight_layout()
        output = plots_dir / f"bitstring_hist_{dataset}.png"
        fig.savefig(output)
        plt.close(fig)
        print(f"Saved {output}")


def plot_energy_makespan_correlation(
    grouped: dict[str, list[dict]], plots_dir: Path, datasets_dir: Path, cache: dict[str, DatasetArtifacts]
) -> None:
    for dataset, entries in grouped.items():
        aggregate = Counter()
        artifacts = None
        for entry in entries:
            artifacts = artifacts or _ensure_artifacts(entry, datasets_dir, cache)
            counts = (entry.get("qaoa") or {}).get("counts") or {}
            aggregate.update({bit.replace(" ", ""): int(freq) for bit, freq in counts.items()})
        if not aggregate or artifacts is None:
            print(f"[advanced_plots] Skipping correlation plot for {dataset} (missing counts/Q).")
            continue

        total = sum(aggregate.values())
        energies = []
        makespans = []
        probs = []
        for bit, freq in aggregate.items():
            probability = freq / total
            energies.append(_bitstring_energy(bit, artifacts.qubo))
            makespans.append(_bitstring_makespan(bit, artifacts.tasks))
            probs.append(probability)

        plt.figure(figsize=(8.5, 5.5))
        sc = plt.scatter(energies, makespans, c=probs, cmap="viridis", s=140, edgecolor="black", linewidth=0.5)
        plt.colorbar(sc, label="Sample probability")
        plt.xlabel("Energy")
        plt.ylabel("Makespan")
        plt.title(f"Energy vs makespan correlation — {dataset}")
        plt.tight_layout()
        output = plots_dir / f"correlation_energy_makespan_{dataset}.png"
        plt.savefig(output)
        plt.close()
        print(f"Saved {output}")


def plot_full_energy_landscape(
    grouped: dict[str, list[dict]],
    plots_dir: Path,
    datasets_dir: Path,
    cache: dict[str, DatasetArtifacts],
    max_qubits: int,
) -> None:
    for dataset, entries in grouped.items():
        artifacts = None
        counts = Counter()
        for entry in entries:
            artifacts = artifacts or _ensure_artifacts(entry, datasets_dir, cache)
            counts.update({bit.replace(" ", ""): int(freq) for bit, freq in (entry.get("qaoa") or {}).get("counts", {}).items()})
        if artifacts is None:
            print(f"[advanced_plots] Skipping full landscape for {dataset} (missing artifacts).")
            continue

        n = artifacts.qubo.shape[0]
        if n > max_qubits:
            print(f"[advanced_plots] Skipping {dataset} landscape (n={n} > limit {max_qubits}).")
            continue

        energies = []
        sample_positions = []
        for idx, bits in enumerate(product("01", repeat=n)):
            bitstring = "".join(bits)
            energy = _bitstring_energy(bitstring, artifacts.qubo)
            energies.append((idx, energy))
            if bitstring in counts:
                sample_positions.append((idx, energy, counts[bitstring]))

        xs, ys = zip(*energies)
        plt.figure(figsize=(10, 5.5))
        plt.scatter(xs, ys, s=12, alpha=0.7, label="All configurations")
        if sample_positions:
            sample_xs = [pos[0] for pos in sample_positions]
            sample_ys = [pos[1] for pos in sample_positions]
            sample_sizes = [40 + 80 * (pos[2] / max(counts.values())) for pos in sample_positions]
            plt.scatter(
                sample_xs,
                sample_ys,
                s=sample_sizes,
                color="#d62728",
                label="Sampled configurations",
                edgecolor="black",
            )
        plt.xlabel("Configuration index")
        plt.ylabel("Energy")
        plt.title(f"Full energy landscape — {dataset} (n={n})")
        plt.legend()
        plt.tight_layout()
        output = plots_dir / f"full_landscape_{dataset}.png"
        plt.savefig(output)
        plt.close()
        print(f"Saved {output}")


def main() -> None:
    args = parse_args()
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(args.results_dir)
    grouped = defaultdict(list)
    for entry in results:
        grouped[_dataset_stem(entry)].append(entry)

    artifact_cache: dict[str, DatasetArtifacts] = {}

    plot_optimizer_convergence(grouped, args.plots_dir)
    plot_depth_scaling(grouped, args.plots_dir)
    plot_classical_vs_qaoa(grouped, args.plots_dir)
    plot_bitstring_histograms(grouped, args.plots_dir, args.datasets_dir, artifact_cache)
    plot_energy_makespan_correlation(grouped, args.plots_dir, args.datasets_dir, artifact_cache)
    plot_full_energy_landscape(
        grouped, args.plots_dir, args.datasets_dir, artifact_cache, args.max_landscape_qubits
    )


if __name__ == "__main__":
    main()
