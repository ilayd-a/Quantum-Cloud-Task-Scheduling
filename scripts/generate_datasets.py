from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from quantum_scheduler.utils import generate_dataset


def main():
    datasets_dir = ROOT / "data" / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    for size in (5, 8, 10, 12):
        generate_dataset(size, str(datasets_dir / f"dataset_{size}.csv"))

    print(f"Datasets written to {datasets_dir}")


if __name__ == "__main__":
    main()
