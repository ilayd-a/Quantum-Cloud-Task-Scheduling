"""Plot helper for experiment outputs produced by run_experiments.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


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


def plot_classical(data: list[dict], plots_dir: Path) -> None:
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


def plot_qaoa_energy(data: list[dict], plots_dir: Path) -> None:
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


def main() -> None:
    args = parse_args()
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    data = load_result_files(args.results_dir)
    plot_classical(data, args.plots_dir)
    plot_qaoa_energy(data, args.plots_dir)


if __name__ == "__main__":
    main()
