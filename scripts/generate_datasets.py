from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from quantum_scheduler.utils import generate_dataset, generate_dataset_by_type


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic scheduling datasets.")
    parser.add_argument("--sizes", type=int, nargs="+", default=(10, 20, 25), help="Job counts to generate")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    parser.add_argument(
        "--instance-type",
        choices=["uniform", "heavy_tailed", "clustered", "default", "all"],
        default="default",
        help="Instance type to generate (or 'all' for all types)",
    )
    parser.add_argument(
        "--fixed-size",
        type=int,
        default=None,
        help="Generate multiple instance types for fixed size (for instance family analysis)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    datasets_dir = ROOT / "data" / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    if args.fixed_size is not None:
        # Generate instance families for fixed size
        instance_types = ["uniform", "heavy_tailed", "clustered"] if args.instance_type == "all" else [args.instance_type]
        for instance_type in instance_types:
            filename = f"dataset_{args.fixed_size}_{instance_type}.csv"
            filepath = datasets_dir / filename
            generate_dataset_by_type(args.fixed_size, str(filepath), instance_type, seed=args.seed)
    else:
        # Original behavior: generate by sizes
        instance_types = ["uniform", "heavy_tailed", "clustered"] if args.instance_type == "all" else [args.instance_type]
        for instance_type in instance_types:
            suffix = f"_{instance_type}" if instance_type != "default" else ""
            for offset, size in enumerate(args.sizes):
                filename = f"dataset_{size}{suffix}.csv"
                filepath = datasets_dir / filename
                generate_dataset_by_type(size, str(filepath), instance_type, seed=args.seed + offset)

    print(f"Datasets written to {datasets_dir}")


if __name__ == "__main__":
    main()
