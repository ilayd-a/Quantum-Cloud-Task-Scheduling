"""Plot helper for experiment outputs produced by run_experiments.py."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from quantum_scheduler.utils.qubo_builder import qubo_from_tasks  # noqa: E402
from quantum_scheduler.utils.decoder import (  # noqa: E402
    bitstring_to_vector,
    decode_solution_vector,
)


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
    try:
        with dataset_path.open() as fh:
            reader = csv.DictReader(fh)
            tasks: list[dict] = []
            for idx, row in enumerate(reader, start=1):
                duration = (row.get("p_i") or row.get("p") or row.get("duration") or "").strip()
                if duration is None:
                    continue
                try:
                    p_i = float(duration)
                except (TypeError, ValueError):
                    continue
                priority_raw = (row.get("priority_w") or row.get("weight") or "").strip()
                try:
                    priority = float(priority_raw) if priority_raw else 1.0
                except (TypeError, ValueError):
                    priority = 1.0
                task = {
                    "job": row.get("job") or row.get("task") or idx,
                    "p_i": p_i,
                    "priority_w": priority,
                }
                tasks.append(task)
        if not tasks:
            raise ValueError("dataset contained zero valid tasks")
        return tasks
    except FileNotFoundError:
        print(f"[plot_results] Dataset missing: {dataset_path}")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[plot_results] Failed to parse {dataset_path}: {exc}")
    return None


def _safe_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_makespan(entry: dict, solver: str) -> float | None:
    metrics = entry.get("metrics") or {}
    candidates: list[float | None] = []
    if solver == "qaoa":
        candidates.extend(
            [
                metrics.get("qaoa_makespan"),
                metrics.get("quantum_makespan"),
                metrics.get("makespan_qaoa"),
                (entry.get("qaoa") or {}).get("best_schedule", {}).get("makespan"),
                (entry.get("qaoa") or {}).get("makespan"),
            ]
        )
    else:
        candidates.extend(
            [
                metrics.get("classical_makespan"),
                metrics.get("baseline_makespan"),
                (entry.get("classical") or {}).get("makespan"),
            ]
        )
    for candidate in candidates:
        converted = _safe_float(candidate)
        if converted is not None:
            return converted
    return None


def _extract_qaoa_energy(entry: dict) -> float | None:
    qaoa = entry.get("qaoa") or {}
    candidates = [
        qaoa.get("energy"),
        qaoa.get("best_sample_energy"),
        qaoa.get("final_energy"),
        (entry.get("metrics") or {}).get("qaoa_energy"),
    ]
    for candidate in candidates:
        converted = _safe_float(candidate)
        if converted is not None:
            return converted
    return None


def _extract_relative_gap(entry: dict) -> float | None:
    metrics = entry.get("metrics") or {}
    gap = metrics.get("relative_gap_pct")
    converted = _safe_float(gap)
    if converted is not None:
        return converted
    qaoa_ms = _extract_makespan(entry, "qaoa")
    classical_ms = _extract_makespan(entry, "classical")
    if qaoa_ms is not None and classical_ms not in (None, 0):
        return 100.0 * (qaoa_ms - classical_ms) / classical_ms
    return None


def _summary_by_size(data: list[dict], value_fn) -> list[dict]:
    buckets: dict[int, list[float]] = {}
    for entry in data:
        size = entry.get("num_tasks")
        value = value_fn(entry)
        if size is None or value is None:
            continue
        buckets.setdefault(int(size), []).append(float(value))

    stats: list[dict] = []
    for size, values in buckets.items():
        arr = np.asarray(values, dtype=float)
        std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        stats.append(
            {
                "num_tasks": size,
                "mean": float(arr.mean()),
                "std": std,
                "min": float(arr.min()),
                "max": float(arr.max()),
                "count": arr.size,
            }
        )
    stats.sort(key=lambda item: item["num_tasks"])
    return stats


def _raw_points(data: list[dict], value_fn) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for entry in data:
        size = entry.get("num_tasks")
        value = value_fn(entry)
        if size is None or value is None:
            continue
        xs.append(int(size))
        ys.append(float(value))
    return xs, ys


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
    stats = _summary_by_size(data, lambda entry: _extract_makespan(entry, "classical"))
    if not stats:
        print("[plot_results] No classical makespans found; skipping trend plot.")
        return None

    sizes = [item["num_tasks"] for item in stats]
    means = [item["mean"] for item in stats]
    mins = [item["min"] for item in stats]
    maxs = [item["max"] for item in stats]
    errs = [item["std"] / np.sqrt(item["count"]) if item["count"] > 1 else 0.0 for item in stats]

    raw_x, raw_y = _raw_points(data, lambda entry: _extract_makespan(entry, "classical"))

    plt.figure(figsize=(9, 5))
    plt.fill_between(sizes, mins, maxs, color="#bcd2f6", alpha=0.35, label="min/max envelope")
    plt.errorbar(
        sizes,
        means,
        yerr=errs,
        fmt="-o",
        color="#1f77b4",
        capsize=4,
        label="mean ± SEM",
    )
    if raw_x:
        plt.scatter(
            raw_x,
            raw_y,
            color="#1f77b4",
            edgecolor="white",
            linewidth=0.4,
            alpha=0.6,
            s=30,
            label="individual runs",
        )

    plt.title("Classical makespan scaling")
    plt.xlabel("Number of jobs")
    plt.ylabel("Makespan (time units)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output_path = plots_dir / "classical_makespans.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")
    return output_path


def plot_qaoa_energy(data: list[dict], plots_dir: Path) -> Path | None:
    stats = _summary_by_size(data, _extract_qaoa_energy)
    if not stats:
        print("[plot_results] No QAOA energies found; skipping energy plot.")
        return None

    sizes = [item["num_tasks"] for item in stats]
    means = [item["mean"] for item in stats]
    errs = [item["std"] / np.sqrt(item["count"]) if item["count"] > 1 else 0.0 for item in stats]
    mins = [item["min"] for item in stats]
    maxs = [item["max"] for item in stats]
    raw_x, raw_y = _raw_points(data, _extract_qaoa_energy)

    plt.figure(figsize=(9, 5))
    plt.fill_between(sizes, mins, maxs, color="#d7b4e6", alpha=0.3, label="min/max envelope")
    plt.errorbar(
        sizes,
        means,
        yerr=errs,
        fmt="-o",
        color="#7a1fa2",
        capsize=4,
        label="mean ± SEM",
    )
    if raw_x:
        plt.scatter(
            raw_x,
            raw_y,
            color="#7a1fa2",
            edgecolor="white",
            linewidth=0.4,
            alpha=0.6,
            s=28,
            label="individual runs",
        )

    plt.title("QAOA energy vs number of jobs")
    plt.xlabel("Number of jobs")
    plt.ylabel("Energy expectation value")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output_path = plots_dir / "qaoa_energy.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")
    return output_path


def plot_makespan_comparison(data: list[dict], plots_dir: Path) -> Path | None:
    grouped: dict[str, dict[str, list[float] | int]] = {}
    for entry in data:
        dataset_label = _dataset_stem(entry)
        qaoa_ms = _extract_makespan(entry, "qaoa")
        classical_ms = _extract_makespan(entry, "classical")
        if qaoa_ms is None or classical_ms is None:
            continue

        bucket = grouped.setdefault(
            dataset_label,
            {"qaoa": [], "classical": [], "num_tasks": entry.get("num_tasks") or 0},
        )
        bucket["qaoa"].append(qaoa_ms)
        bucket["classical"].append(classical_ms)
        bucket["num_tasks"] = entry.get("num_tasks") or bucket["num_tasks"]

    if not grouped:
        print("[plot_results] No makespan metrics available; skipping bar chart.")
        return None

    summaries = []
    for dataset, values in grouped.items():
        q_arr = np.asarray(values["qaoa"], dtype=float)
        c_arr = np.asarray(values["classical"], dtype=float)
        summaries.append(
            {
                "dataset": dataset,
                "num_tasks": values["num_tasks"],
                "qaoa_mean": float(q_arr.mean()),
                "classical_mean": float(c_arr.mean()),
                "qaoa_sem": float(q_arr.std(ddof=1) / np.sqrt(q_arr.size)) if q_arr.size > 1 else 0.0,
                "classical_sem": float(c_arr.std(ddof=1) / np.sqrt(c_arr.size)) if c_arr.size > 1 else 0.0,
            }
        )

    summaries.sort(key=lambda item: (item["num_tasks"], item["dataset"]))
    labels = [f"{item['dataset']} (n={item['num_tasks']})" for item in summaries]
    q_means = [item["qaoa_mean"] for item in summaries]
    c_means = [item["classical_mean"] for item in summaries]
    q_errors = [item["qaoa_sem"] for item in summaries]
    c_errors = [item["classical_sem"] for item in summaries]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(max(10, len(labels) * 1.2), 5.5))
    plt.bar(
        x - width / 2,
        c_means,
        width,
        yerr=c_errors,
        capsize=4,
        label="Classical",
        color="#4c72b0",
        alpha=0.85,
    )
    plt.bar(
        x + width / 2,
        q_means,
        width,
        yerr=q_errors,
        capsize=4,
        label="QAOA",
        color="#dd8452",
        alpha=0.9,
    )

    for idx, (q_val, c_val) in enumerate(zip(q_means, c_means)):
        if c_val:
            gap = 100.0 * (q_val - c_val) / c_val
            plt.text(
                x[idx],
                max(q_val, c_val) * 1.01,
                f"{gap:+.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#333333",
            )

    plt.title("Makespan comparison (mean ± SEM)")
    plt.ylabel("Makespan (time units)")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.legend()
    plt.tight_layout()
    output_path = plots_dir / "makespan_comparison.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")
    return output_path


def _extract_energy_trace(entry: dict) -> list[dict]:
    candidates = [
        (entry.get("qaoa") or {}).get("energy_trace"),
        (entry.get("metrics") or {}).get("energy_trace"),
        entry.get("energy_trace"),
    ]
    for candidate in candidates:
        if candidate:
            return candidate
    return []


def plot_convergence_curves(data: list[dict], plots_dir: Path) -> Path | None:
    plt.figure(figsize=(10, 6))
    plotted = False
    for entry in data:
        trace = _extract_energy_trace(entry)
        if not trace:
            continue
        restarts: dict[int, list[dict]] = {}
        for point in trace:
            restart = int(point.get("restart") or 1)
            if point.get("energy") is None:
                continue
            restarts.setdefault(restart, []).append(point)
        for restart, points in restarts.items():
            if not points:
                continue
            points.sort(key=lambda pt: pt.get("evaluation") or pt.get("iteration") or pt.get("step") or 0)
            energies = np.array([float(pt.get("energy")) for pt in points], dtype=float)
            if energies.size == 0:
                continue
            running_best = np.minimum.accumulate(energies)
            evals = [
                int(pt.get("evaluation") or pt.get("iteration") or pt.get("step") or idx + 1)
                for idx, pt in enumerate(points)
            ]
            label = f"{_dataset_stem(entry)} r{restart}"
            plt.plot(evals[: running_best.size], running_best, linewidth=1.4, label=label)
            plotted = True

    if not plotted:
        plt.close()
        print("[plot_results] No convergence traces found; skipping convergence curve.")
        return None

    plt.title("Best-seen QAOA energy vs evaluations")
    plt.xlabel("Energy evaluation index")
    plt.ylabel("Best energy so far")
    plt.grid(alpha=0.3, linestyle="--", linewidth=0.6)
    plt.legend(fontsize=8)
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
    percentiles = np.linspace(0, 100, len(sorted_energies))
    median = np.percentile(sorted_energies, 50)
    q1, q3 = np.percentile(sorted_energies, [25, 75])

    plt.figure(figsize=(8.5, 5))
    plt.plot(percentiles, sorted_energies, color="#55a868", linewidth=1.8)
    plt.fill_between(
        percentiles,
        q1,
        q3,
        color="#55a868",
        alpha=0.12,
        label="interquartile band",
    )
    plt.axhline(median, color="#2a7d46", linestyle="--", linewidth=0.9, label="median energy")
    plt.title(f"QUBO energy landscape ({_dataset_stem(entry)})")
    plt.xlabel("Energy percentile (%)")
    plt.ylabel("Energy")
    plt.legend()
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
    vmax = float(np.max(np.abs(Q)))
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    plt.figure(figsize=(6.5, 5.5))
    im = plt.imshow(Q, cmap="coolwarm", aspect="auto", norm=norm)
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

    tasks = _load_tasks(entry) or []
    durations = [float(task["p_i"]) for task in tasks]
    makespans: list[float | None] = []
    if durations:
        for bitstring, _ in top_items:
            vector = bitstring_to_vector(bitstring, len(durations))
            schedule = decode_solution_vector(vector, durations)
            makespans.append(float(schedule["makespan"]))
    else:
        makespans = [None] * len(labels)

    fig, ax_prob = plt.subplots(figsize=(max(9, len(labels)), 4.5))
    ax_prob.bar(x, probs, color="#8172b2", alpha=0.85, label="Sample probability")
    ax_prob.set_title(f"Top-{len(labels)} QAOA samples ({_dataset_stem(entry)})")
    ax_prob.set_ylabel("Probability")
    ax_prob.set_xticks(x, labels)
    ax_prob.set_xticklabels(labels, rotation=40, ha="right")

    added_secondary = False
    if any(ms is not None for ms in makespans):
        ax_ms = ax_prob.twinx()
        ms_vals = [ms if ms is not None else np.nan for ms in makespans]
        ax_ms.plot(
            x,
            ms_vals,
            color="#c44e52",
            marker="s",
            linewidth=1.2,
            label="Implied makespan",
        )
        ax_ms.set_ylabel("Makespan (time units)")
        added_secondary = True

    handles, labels_leg = ax_prob.get_legend_handles_labels()
    if added_secondary:
        handles2, labels2 = ax_ms.get_legend_handles_labels()
        handles += handles2
        labels_leg += labels2
    ax_prob.legend(handles, labels_leg, loc="upper right")
    fig.tight_layout()
    output_path = plots_dir / f"{_dataset_stem(entry)}_sample_histogram.png"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved {output_path}")
    return output_path


def plot_relative_gap(data: list[dict], plots_dir: Path) -> Path | None:
    stats = _summary_by_size(data, _extract_relative_gap)
    if not stats:
        print("[plot_results] No relative gap metrics; skipping gap plot.")
        return None

    sizes = [item["num_tasks"] for item in stats]
    means = [item["mean"] for item in stats]
    errs = [item["std"] / np.sqrt(item["count"]) if item["count"] > 1 else 0.0 for item in stats]
    mins = [item["min"] for item in stats]
    maxs = [item["max"] for item in stats]

    plt.figure(figsize=(9, 5))
    plt.fill_between(sizes, mins, maxs, color="#f5c4a6", alpha=0.35, label="min/max envelope")
    plt.errorbar(
        sizes,
        means,
        yerr=errs,
        fmt="-o",
        color="#c44e52",
        capsize=4,
        label="mean ± SEM",
    )
    plt.axhline(0, color="#333333", linestyle="--", linewidth=0.9, label="Parity line")
    plt.title("Relative makespan gap (QAOA vs classical)")
    plt.ylabel("Gap (%)  (negative = QAOA better)")
    plt.xlabel("Number of jobs")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output_path = plots_dir / "relative_gap.png"
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
    plot_relative_gap(data, args.plots_dir)
    plot_convergence_curves(data, args.plots_dir)

    for entry in data:
        plot_qubo_landscape(entry, args.plots_dir)
        plot_qubo_heatmap(entry, args.plots_dir)
        plot_sample_histogram(entry, args.plots_dir)


if __name__ == "__main__":
    main()
