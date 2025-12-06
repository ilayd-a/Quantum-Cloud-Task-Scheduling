from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from quantum_scheduler.utils import generate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic scheduling datasets.")
    parser.add_argument("--sizes", type=int, nargs="+", default=(5, 8, 10, 12), help="Job counts to generate")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    return parser.parse_args()


def main():
    args = parse_args()
    datasets_dir = ROOT / "data" / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    for offset, size in enumerate(args.sizes):
        filepath = datasets_dir / f"dataset_{size}.csv"
        generate_dataset(size, str(filepath), seed=args.seed + offset)

    print(f"Datasets written to {datasets_dir}")


if __name__ == "__main__":
    main()
