"""Render a QAOA circuit diagram for a given dataset/config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qiskit.visualization import circuit_drawer

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from quantum_scheduler.qaoa_solver import qaoa_circuit, qubo_to_ising  # noqa: E402
from quantum_scheduler.utils import load_tasks  # noqa: E402
from quantum_scheduler.utils.qubo_builder import qubo_from_tasks  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="dataset_5.csv",
        help="Dataset filename under data/datasets/ or absolute path",
    )
    parser.add_argument("--reps", type=int, default=1, help="QAOA depth p")
    parser.add_argument(
        "--balance-penalty",
        type=float,
        default=10.0,
        help="Penalty multiplier A used when constructing the QUBO",
    )
    parser.add_argument(
        "--priority-bias",
        type=float,
        default=0.1,
        help="Linear bias encouraging high-priority jobs on machine 1",
    )
    parser.add_argument(
        "--drawer",
        choices=["mpl", "text"],
        default="mpl",
        help="Rendering backend: matplotlib figure or ascii text",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output resolution when using the matplotlib drawer",
    )
    parser.add_argument(
        "--measure",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include measurement operations at the end of the circuit",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit output path. Defaults to results/circuit_diagrams/<dataset>_p<reps>.png",
    )
    return parser.parse_args()


def resolve_dataset_path(dataset_arg: str | Path) -> Path:
    path = Path(dataset_arg)
    if not path.is_absolute():
        path = ROOT / "data" / "datasets" / path
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return path


def build_qaoa_circuit(
    dataset_path: Path,
    reps: int,
    balance_penalty: float,
    priority_bias: float,
    measure: bool,
):
    tasks = load_tasks(dataset_path)
    Q, *_ = qubo_from_tasks(
        tasks,
        balance_penalty_multiplier=balance_penalty,
        priority_bias=priority_bias,
    )
    h, J = qubo_to_ising(Q)
    qc, *_ = qaoa_circuit(h, J, len(tasks), reps=reps, measure=measure)
    return qc


def main() -> None:
    args = parse_args()
    dataset_path = resolve_dataset_path(args.dataset)
    qc = build_qaoa_circuit(
        dataset_path=dataset_path,
        reps=args.reps,
        balance_penalty=args.balance_penalty,
        priority_bias=args.priority_bias,
        measure=args.measure,
    )

    default_output = (
        ROOT
        / "results"
        / "circuit_diagrams"
        / f"{dataset_path.stem}_p{args.reps}.{ 'txt' if args.drawer == 'text' else 'png'}"
    )
    output_path = args.output or default_output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.drawer == "text":
        diagram = qc.draw(output="text")
        output_path.write_text(str(diagram))
    else:
        fig = circuit_drawer(qc, output="mpl", fold=-1)
        fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved circuit diagram to {output_path}")


if __name__ == "__main__":
    main()
