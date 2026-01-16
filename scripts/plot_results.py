#!/usr/bin/env python3
"""
Plot helper for experiment outputs produced by run_experiments.py.

- Reads *_results.json under --results-dir
- Optional filtering by --tag:
    * checks entry["tag"], entry["config"]["tag"], entry["metrics"]["tag"]
    * also falls back to matching the tag against the filename/path
- Supports plotting either min-energy decoded schedule or Top-K postselected schedule
  via --qaoa-selection {min-energy, top-k}.

Example:
  python scripts/plot_results.py --results-dir results --tag k200 --qaoa-selection top-k
"""

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
from quantum_scheduler.utils.decoder import bitstring_to_vector, decode_solution_vector  # noqa: E402


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
        default=None,
        type=Path,
        help="Directory where plots will be saved (default: <results-dir>/plots)",
    )
    parser.add_argument(
        "--tag",
        default=None,
        type=str,
        help=(
            "Filter results by tag. Matches entry['tag'] / entry['config']['tag'] if present; "
            "otherwise falls back to matching against filename/path."
        ),
    )
    parser.add_argument(
        "--qaoa-selection",
        choices=["min-energy", "top-k"],
        default="top-k",
        help="Which QAOA solution to plot: min-energy or top-k postselection",
    )
    parser.add_argument(
        "--debug-tag",
        action="store_true",
        help="Print detailed debug output showing why each file was kept or dropped during tag filtering.",
    )
    return parser.parse_args()


def load_result_files(results_dir: Path) -> list[dict]:
    # Search recursively for *_results.json files
    files = sorted(results_dir.rglob("*_results.json"))
    if not files:
        raise FileNotFoundError(
            f"No results found under {results_dir}. Run scripts/run_experiments.py first."
        )

    payloads: list[dict] = []
    for path in files:
        try:
            entry = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            print(f"[plot_results] Skipping invalid JSON: {path} ({exc})")
            continue

        # Keep origin path for filename/path tag filtering + debugging
        entry["_result_path"] = str(path)
        entry["_result_name"] = path.name
        payloads.append(entry)

    payloads.sort(key=lambda entry: entry.get("num_tasks", 0))
    return payloads


def _norm_tag(val) -> str:
    if val is None:
        return ""
    if not isinstance(val, str):
        val = str(val)
    return val.strip().casefold()


def filter_by_tag(data: list[dict], tag: str, debug: bool = False) -> list[dict]:
    """
    Filter entries by tag, checking multiple locations:
    1. entry["tag"]
    2. entry["config"]["tag"]
    3. entry["metrics"]["tag"]
    4. Filename (from _result_name or Path(_result_path).name)
    
    Returns filtered list and optionally prints debug info.
    """
    if not tag:
        return data
    
    want = _norm_tag(tag)
    filtered: list[dict] = []

    for e in data:
        matched = False
        match_reason = None
        
        # 1) Check explicit tag fields
        tag1 = _norm_tag(e.get("tag"))
        tag2 = _norm_tag((e.get("config") or {}).get("tag"))
        tag3 = _norm_tag((e.get("metrics") or {}).get("tag"))

        if want == tag1:
            matched = True
            match_reason = "entry['tag']"
        elif want == tag2:
            matched = True
            match_reason = "entry['config']['tag']"
        elif want == tag3:
            matched = True
            match_reason = "entry['metrics']['tag']"
        else:
            # 2) Filename/path fallback (case-insensitive substring match)
            # Try _result_name first (should be Path.name)
            filename = e.get("_result_name")
            if filename:
                n = _norm_tag(filename)
                if want in n:
                    matched = True
                    match_reason = f"filename (from _result_name: {filename})"
            
            # Also check _result_path if _result_name wasn't set
            if not matched:
                result_path = e.get("_result_path")
                if result_path:
                    # Extract just the filename from the path
                    path_obj = Path(result_path)
                    path_filename = _norm_tag(path_obj.name)
                    if want in path_filename:
                        matched = True
                        match_reason = f"filename (from Path(_result_path).name: {path_obj.name})"
                    # Also check full path as fallback
                    elif want in _norm_tag(result_path):
                        matched = True
                        match_reason = f"path substring (from _result_path: {result_path})"
        
        if matched:
            filtered.append(e)
            if debug:
                filename_display = e.get("_result_name") or Path(e.get("_result_path", "")).name or "unknown"
                print(f"[plot_results] KEPT: {filename_display} (matched via {match_reason})")
        elif debug:
            filename_display = e.get("_result_name") or Path(e.get("_result_path", "")).name or "unknown"
            tag1_val = e.get('tag')
            tag2_val = (e.get('config') or {}).get('tag')
            tag3_val = (e.get('metrics') or {}).get('tag')
            # Show normalized filename for debugging
            normalized_filename = _norm_tag(filename_display)
            print(f"[plot_results] DROPPED: {filename_display} (normalized: '{normalized_filename}', searching for '{want}', found: entry.tag={tag1_val}, config.tag={tag2_val}, metrics.tag={tag3_val})")

    return filtered


def _dataset_stem(entry: dict) -> str:
    dataset = entry.get("dataset") or Path(entry.get("dataset_path", "dataset")).name
    return Path(dataset).stem


def _resolve_dataset_path(entry: dict) -> Path | None:
    path = entry.get("dataset_path")
    if path:
        resolved = Path(path)
        if resolved.exists():
            return resolved

    fallback = ROOT / "data" / "datasets" / (entry.get("dataset") or "")
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
                duration = row.get("p_i") or row.get("p") or row.get("duration") or ""
                duration = duration.strip() if isinstance(duration, str) else duration
                if duration is None or duration == "":
                    continue

                try:
                    p_i = float(duration)
                except (TypeError, ValueError):
                    continue

                priority_raw = row.get("priority_w") or row.get("weight") or ""
                priority_raw = priority_raw.strip() if isinstance(priority_raw, str) else priority_raw
                try:
                    priority = float(priority_raw) if priority_raw else 1.0
                except (TypeError, ValueError):
                    priority = 1.0

                tasks.append(
                    {
                        "job": row.get("job") or row.get("task") or idx,
                        "p_i": p_i,
                        "priority_w": priority,
                    }
                )

        if not tasks:
            raise ValueError("dataset contained zero valid tasks")
        return tasks

    except FileNotFoundError:
        print(f"[plot_results] Dataset missing: {dataset_path}")
    except Exception as exc:
        print(f"[plot_results] Failed to parse {dataset_path}: {exc}")

    return None


def _safe_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_makespan(entry: dict, solver: str, qaoa_selection: str = "top-k") -> float | None:
    metrics = entry.get("metrics") or {}

    if solver == "qaoa":
        if qaoa_selection == "min-energy":
            # Min-energy selection: prioritize qaoa_makespan (from best_schedule decoded from min-energy bitstring)
            candidates = [
                metrics.get("qaoa_makespan"),  # This is the min-energy selection makespan
                (entry.get("qaoa") or {}).get("best_schedule", {}).get("makespan"),  # Fallback to best_schedule
                (entry.get("qaoa") or {}).get("makespan"),  # Another fallback
                metrics.get("quantum_makespan"),
                metrics.get("makespan_qaoa"),
                (entry.get("makespans") or {}).get("min_energy"),
                (entry.get("makespans") or {}).get("min-energy"),
                metrics.get("qaoa_makespan_min_energy"),
                metrics.get("qaoa_makespan_min-energy"),
            ]
        else:
            # Top-K postselection: prioritize topk_makespan
            candidates = [
                metrics.get("topk_makespan"),  # This is the top-K postselection makespan
                (entry.get("qaoa") or {}).get("topk_schedule", {}).get("makespan"),  # Fallback to topk_schedule
                metrics.get("qaoa_makespan_topk"),
                metrics.get("qaoa_makespan_top_k"),
                metrics.get("qaoa_makespan_top-k"),
                (entry.get("makespans") or {}).get("top_k"),
                (entry.get("makespans") or {}).get("top-k"),
                # Don't fall back to qaoa_makespan for top-k selection - it's different!
            ]

        for cand in candidates:
            v = _safe_float(cand)
            if v is not None:
                return v
        return None

    # classical
    candidates = [
        (entry.get("makespans") or {}).get("classical"),
        metrics.get("classical_makespan"),
        metrics.get("baseline_makespan"),
        (entry.get("classical") or {}).get("makespan"),
    ]
    for cand in candidates:
        v = _safe_float(cand)
        if v is not None:
            return v
    return None


def _extract_qaoa_energy(entry: dict) -> float | None:
    qaoa = entry.get("qaoa") or {}
    candidates = [
        qaoa.get("energy"),
        qaoa.get("best_sample_energy"),
        qaoa.get("final_energy"),
        (entry.get("metrics") or {}).get("qaoa_energy"),
    ]
    for cand in candidates:
        v = _safe_float(cand)
        if v is not None:
            return v
    return None


def _extract_relative_gap(entry: dict, qaoa_selection: str) -> float | None:
    # Always recompute from the chosen qaoa_selection first (min-energy vs top-k)
    q_ms = _extract_makespan(entry, "qaoa", qaoa_selection=qaoa_selection)
    c_ms = _extract_makespan(entry, "classical")
    if q_ms is not None and c_ms not in (None, 0):
        return 100.0 * (q_ms - c_ms) / c_ms

    # fallback to stored field if present
    gap = (entry.get("metrics") or {}).get("relative_gap_pct")
    v = _safe_float(gap)
    return v


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
                "num_tasks": int(size),
                "mean": float(arr.mean()),
                "std": std,
                "min": float(arr.min()),
                "max": float(arr.max()),
                "count": int(arr.size),
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
    qaoa_cfg = entry.get("qaoa", {}) or {}
    config = entry.get("config", {}) or {}

    penalty = qaoa_cfg.get("balance_penalty_multiplier") or config.get("balance_penalty_multiplier")
    priority = qaoa_cfg.get("priority_bias") or config.get("priority_bias", 0.1)

    Q, *_ = qubo_from_tasks(
        tasks,
        balance_penalty_multiplier=penalty,
        priority_bias=priority,
    )
    return Q


def plot_classical(data: list[dict], plots_dir: Path) -> None:
    stats = _summary_by_size(data, lambda e: _extract_makespan(e, "classical"))
    if not stats:
        print("[plot_results] No classical makespans found; skipping classical trend plot.")
        return

    sizes = [s["num_tasks"] for s in stats]
    means = [s["mean"] for s in stats]
    mins = [s["min"] for s in stats]
    maxs = [s["max"] for s in stats]
    errs = [s["std"] / np.sqrt(s["count"]) if s["count"] > 1 else 0.0 for s in stats]
    raw_x, raw_y = _raw_points(data, lambda e: _extract_makespan(e, "classical"))

    plt.figure(figsize=(9, 5))
    plt.fill_between(sizes, mins, maxs, alpha=0.25, label="min/max envelope")
    plt.errorbar(sizes, means, yerr=errs, fmt="-o", capsize=4, label="mean ± SEM")
    if raw_x:
        plt.scatter(raw_x, raw_y, edgecolor="white", linewidth=0.4, alpha=0.6, s=30, label="individual runs")

    plt.title("Classical makespan scaling")
    plt.xlabel("Number of jobs")
    plt.ylabel("Makespan (time units)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out = plots_dir / "classical_makespans.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


def plot_qaoa_energy(data: list[dict], plots_dir: Path) -> None:
    stats = _summary_by_size(data, _extract_qaoa_energy)
    if not stats:
        print("[plot_results] No QAOA energies found; skipping energy plot.")
        return

    sizes = [s["num_tasks"] for s in stats]
    means = [s["mean"] for s in stats]
    mins = [s["min"] for s in stats]
    maxs = [s["max"] for s in stats]
    errs = [s["std"] / np.sqrt(s["count"]) if s["count"] > 1 else 0.0 for s in stats]
    raw_x, raw_y = _raw_points(data, _extract_qaoa_energy)

    plt.figure(figsize=(9, 5))
    plt.fill_between(sizes, mins, maxs, alpha=0.25, label="min/max envelope")
    plt.errorbar(sizes, means, yerr=errs, fmt="-o", capsize=4, label="mean ± SEM")
    if raw_x:
        plt.scatter(raw_x, raw_y, edgecolor="white", linewidth=0.4, alpha=0.6, s=28, label="individual runs")

    plt.title("QAOA energy vs number of jobs")
    plt.xlabel("Number of jobs")
    plt.ylabel("Energy expectation value")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out = plots_dir / "qaoa_energy.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


def plot_relative_gap(data: list[dict], plots_dir: Path, qaoa_selection: str) -> None:
    stats = _summary_by_size(data, lambda e: _extract_relative_gap(e, qaoa_selection))
    if not stats:
        print("[plot_results] No relative gap metrics; skipping gap plot.")
        return

    sizes = [s["num_tasks"] for s in stats]
    means = [s["mean"] for s in stats]
    mins = [s["min"] for s in stats]
    maxs = [s["max"] for s in stats]
    errs = [s["std"] / np.sqrt(s["count"]) if s["count"] > 1 else 0.0 for s in stats]

    plt.figure(figsize=(9, 5))
    plt.fill_between(sizes, mins, maxs, alpha=0.25, label="min/max envelope")
    plt.errorbar(sizes, means, yerr=errs, fmt="-o", capsize=4, label="mean ± SEM")
    plt.axhline(0, linestyle="--", linewidth=0.9, label="Parity line")
    plt.title(f"Relative makespan gap (QAOA {qaoa_selection} vs classical)")
    plt.ylabel("Gap (%) (negative = QAOA better)")
    plt.xlabel("Number of jobs")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out = plots_dir / "relative_gap.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


def plot_makespan_comparison(data: list[dict], plots_dir: Path, qaoa_selection: str) -> None:
    grouped: dict[str, dict[str, list[float] | int]] = {}
    for entry in data:
        label = _dataset_stem(entry)
        q_ms = _extract_makespan(entry, "qaoa", qaoa_selection=qaoa_selection)
        c_ms = _extract_makespan(entry, "classical")
        if q_ms is None or c_ms is None:
            continue
        bucket = grouped.setdefault(label, {"qaoa": [], "classical": [], "num_tasks": int(entry.get("num_tasks") or 0)})
        bucket["qaoa"].append(q_ms)
        bucket["classical"].append(c_ms)
        bucket["num_tasks"] = int(entry.get("num_tasks") or bucket["num_tasks"])

    if not grouped:
        print("[plot_results] No makespan metrics available; skipping bar chart.")
        return

    summaries = []
    for dataset, values in grouped.items():
        q_arr = np.asarray(values["qaoa"], dtype=float)
        c_arr = np.asarray(values["classical"], dtype=float)
        summaries.append(
            {
                "dataset": dataset,
                "num_tasks": int(values["num_tasks"]),
                "qaoa_mean": float(q_arr.mean()),
                "classical_mean": float(c_arr.mean()),
                "qaoa_sem": float(q_arr.std(ddof=1) / np.sqrt(q_arr.size)) if q_arr.size > 1 else 0.0,
                "classical_sem": float(c_arr.std(ddof=1) / np.sqrt(c_arr.size)) if c_arr.size > 1 else 0.0,
            }
        )
    summaries.sort(key=lambda it: (it["num_tasks"], it["dataset"]))

    labels = [f"{it['dataset']} (n={it['num_tasks']})" for it in summaries]
    q_means = [it["qaoa_mean"] for it in summaries]
    c_means = [it["classical_mean"] for it in summaries]
    q_err = [it["qaoa_sem"] for it in summaries]
    c_err = [it["classical_sem"] for it in summaries]

    x = np.arange(len(labels))
    w = 0.35

    plt.figure(figsize=(max(10, len(labels) * 1.2), 5.5))
    plt.bar(x - w / 2, c_means, w, yerr=c_err, capsize=4, label="Classical", alpha=0.85)
    plt.bar(x + w / 2, q_means, w, yerr=q_err, capsize=4, label=f"QAOA ({qaoa_selection})", alpha=0.9)

    for i, (qv, cv) in enumerate(zip(q_means, c_means)):
        if cv:
            gap = 100.0 * (qv - cv) / cv
            plt.text(x[i], max(qv, cv) * 1.01, f"{gap:+.1f}%", ha="center", va="bottom", fontsize=8)

    plt.title("Makespan comparison (mean ± SEM)")
    plt.ylabel("Makespan (time units)")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.legend()
    plt.tight_layout()
    out = plots_dir / "makespan_comparison.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


def _extract_energy_trace(entry: dict) -> list[dict]:
    candidates = [
        (entry.get("qaoa") or {}).get("energy_trace"),
        (entry.get("metrics") or {}).get("energy_trace"),
        entry.get("energy_trace"),
    ]
    for c in candidates:
        if c:
            return c
    return []


def plot_convergence_curves(data: list[dict], plots_dir: Path) -> None:
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
            points.sort(key=lambda pt: pt.get("evaluation") or pt.get("iteration") or pt.get("step") or 0)
            energies = np.array([float(pt["energy"]) for pt in points], dtype=float)
            if energies.size == 0:
                continue
            best = np.minimum.accumulate(energies)
            evals = [int(pt.get("evaluation") or pt.get("iteration") or pt.get("step") or (i + 1)) for i, pt in enumerate(points)]
            plt.plot(evals[: best.size], best, linewidth=1.4, label=f"{_dataset_stem(entry)} r{restart}")
            plotted = True

    if not plotted:
        plt.close()
        print("[plot_results] No convergence traces found; skipping convergence curve.")
        return

    plt.title("Best-seen QAOA energy vs evaluations")
    plt.xlabel("Energy evaluation index")
    plt.ylabel("Best energy so far")
    plt.grid(alpha=0.3, linestyle="--", linewidth=0.6)
    plt.legend(fontsize=8)
    plt.tight_layout()
    out = plots_dir / "convergence_curves.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


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
        energies.append(float(vec @ Q @ vec))
    return energies


def plot_qubo_landscape(entry: dict, plots_dir: Path) -> None:
    Q = _compute_qubo_matrix(entry)
    if Q is None:
        return
    energies = _sample_qubo_energies(Q)
    if not energies:
        return

    e = np.sort(np.asarray(energies, dtype=float))
    p = np.linspace(0, 100, len(e))
    med = np.percentile(e, 50)
    q1, q3 = np.percentile(e, [25, 75])

    plt.figure(figsize=(8.5, 5))
    plt.plot(p, e, linewidth=1.8)
    plt.fill_between(p, q1, q3, alpha=0.12, label="interquartile band")
    plt.axhline(med, linestyle="--", linewidth=0.9, label="median energy")
    plt.title(f"QUBO energy landscape ({_dataset_stem(entry)})")
    plt.xlabel("Energy percentile (%)")
    plt.ylabel("Energy")
    plt.legend()
    plt.tight_layout()
    out = plots_dir / f"{_dataset_stem(entry)}_qubo_landscape.png"
    plt.savefig(out)
    plt.close()


def plot_qubo_heatmap(entry: dict, plots_dir: Path) -> None:
    Q = _compute_qubo_matrix(entry)
    if Q is None:
        return
    vmax = float(np.max(np.abs(Q)))
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    plt.figure(figsize=(6.5, 5.5))
    im = plt.imshow(Q, cmap="coolwarm", aspect="auto", norm=norm)
    plt.title(f"QUBO coefficients heatmap ({_dataset_stem(entry)})")
    plt.xlabel("Variable index")
    plt.ylabel("Variable index")
    plt.colorbar(im, label="Q_ij")
    plt.tight_layout()
    out = plots_dir / f"{_dataset_stem(entry)}_qubo_heatmap.png"
    plt.savefig(out)
    plt.close()


def plot_sample_histogram(entry: dict, plots_dir: Path, top_k: int = 12) -> None:
    counts = (entry.get("qaoa") or {}).get("counts") or {}
    if not counts:
        return

    total = sum(counts.values())
    if total <= 0:
        return

    top_items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    labels = [bit for bit, _ in top_items]
    probs = [c / total for _, c in top_items]
    x = np.arange(len(labels))

    tasks = _load_tasks(entry) or []
    durations = [float(t["p_i"]) for t in tasks]

    makespans: list[float | None] = []
    if durations:
        for bitstring, _ in top_items:
            vec = bitstring_to_vector(bitstring, len(durations))
            sched = decode_solution_vector(vec, durations)
            makespans.append(float(sched["makespan"]))
    else:
        makespans = [None] * len(labels)

    fig, ax = plt.subplots(figsize=(max(9, len(labels)), 4.5))
    ax.bar(x, probs, alpha=0.85, label="Sample probability")
    ax.set_title(f"Top-{len(labels)} QAOA samples ({_dataset_stem(entry)})")
    ax.set_ylabel("Probability")
    ax.set_xticks(x, labels)
    ax.set_xticklabels(labels, rotation=40, ha="right")

    if any(ms is not None for ms in makespans):
        ax2 = ax.twinx()
        ms_vals = [ms if ms is not None else np.nan for ms in makespans]
        ax2.plot(x, ms_vals, marker="s", linewidth=1.2, label="Implied makespan")
        ax2.set_ylabel("Makespan (time units)")

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right")
    else:
        ax.legend(loc="upper right")

    fig.tight_layout()
    out = plots_dir / f"{_dataset_stem(entry)}_sample_histogram.png"
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    results_dir = args.results_dir
    plots_dir = args.plots_dir if args.plots_dir is not None else (results_dir / "plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    data = load_result_files(results_dir)

    if args.tag:
        before = len(data)
        if args.debug_tag:
            print(f"[plot_results] DEBUG: Filtering {before} files with tag '{args.tag}'...")
        data_tag = filter_by_tag(data, args.tag, debug=args.debug_tag)
        after = len(data_tag)
        print(f"[plot_results] Tag filter '{args.tag}': {after}/{before} files kept")

        if after == 0:
            print("[plot_results] WARNING: No files matched the tag filter.")
            if args.debug_tag:
                print("[plot_results] DEBUG: showing up to 15 result filenames to help you pick the tag:")
                for e in data[:15]:
                    filename = e.get("_result_name") or Path(e.get("_result_path", "")).name or "unknown"
                    print(f"  - {filename}")
                print("[plot_results] DEBUG: showing example tag fields from first 5 JSONs:")
                for e in data[:5]:
                    filename = e.get("_result_name") or Path(e.get("_result_path", "")).name or "unknown"
                    print(f"    {filename}: entry.tag={e.get('tag')}, config.tag={(e.get('config') or {}).get('tag')}, metrics.tag={(e.get('metrics') or {}).get('tag')}")
            else:
                print("[plot_results] Tip: Use --debug-tag to see why files were dropped.")

        data = data_tag

    if not data:
        raise RuntimeError(
            "No usable results after filtering. "
            "Try running without --tag, or run with --debug-tag to inspect filenames and tag fields."
        )

    plot_classical(data, plots_dir)
    plot_qaoa_energy(data, plots_dir)
    plot_makespan_comparison(data, plots_dir, args.qaoa_selection)
    plot_relative_gap(data, plots_dir, args.qaoa_selection)
    plot_convergence_curves(data, plots_dir)

    for entry in data:
        plot_qubo_landscape(entry, plots_dir)
        plot_qubo_heatmap(entry, plots_dir)
        plot_sample_histogram(entry, plots_dir)


if __name__ == "__main__":
    main()
