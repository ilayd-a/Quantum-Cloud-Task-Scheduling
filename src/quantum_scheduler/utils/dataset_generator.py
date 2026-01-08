import numpy as np
import pandas as pd
from typing import Optional, Literal


def generate_dataset(n: int, file_path: str, seed: Optional[int] = None):
    """Generate a dataset with uniform random job durations (original method)."""
    rng = np.random.default_rng(seed)
    p = rng.integers(5, 20, size=n)
    w = rng.integers(1, 4, size=n)

    df = pd.DataFrame(
        {
            "job": np.arange(1, n + 1),
            "p_i": p,
            "priority_w": w,
        }
    )

    df.to_csv(file_path, index=False)
    print(f"Generated dataset: {file_path} (seed={seed})")


def generate_dataset_uniform(n: int, file_path: str, seed: Optional[int] = None, min_duration: int = 5, max_duration: int = 20):
    """Generate dataset with uniform job durations (all similar sizes)."""
    rng = np.random.default_rng(seed)
    # Uniform distribution: all jobs have similar durations
    p = rng.integers(min_duration, max_duration + 1, size=n)
    w = rng.integers(1, 4, size=n)

    df = pd.DataFrame(
        {
            "job": np.arange(1, n + 1),
            "p_i": p,
            "priority_w": w,
        }
    )

    df.to_csv(file_path, index=False)
    print(f"Generated uniform dataset: {file_path} (seed={seed}, mean={p.mean():.1f})")


def generate_dataset_heavy_tailed(n: int, file_path: str, seed: Optional[int] = None, min_duration: int = 5, max_duration: int = 100):
    """Generate dataset with heavy-tailed job durations (few very large jobs)."""
    rng = np.random.default_rng(seed)
    # Heavy-tailed: exponential-like distribution
    # Mix of small (5-15) and occasional very large (50-100) jobs
    p = np.zeros(n, dtype=int)
    for i in range(n):
        if rng.random() < 0.2:  # 20% chance of large job
            p[i] = rng.integers(50, max_duration + 1)
        else:
            p[i] = rng.integers(min_duration, 16)
    
    w = rng.integers(1, 4, size=n)

    df = pd.DataFrame(
        {
            "job": np.arange(1, n + 1),
            "p_i": p,
            "priority_w": w,
        }
    )

    df.to_csv(file_path, index=False)
    print(f"Generated heavy-tailed dataset: {file_path} (seed={seed}, mean={p.mean():.1f}, max={p.max()})")


def generate_dataset_clustered(n: int, file_path: str, seed: Optional[int] = None, n_large: int | None = None):
    """Generate clustered workload: few large jobs, many small jobs."""
    rng = np.random.default_rng(seed)
    
    if n_large is None:
        n_large = max(1, n // 5)  # ~20% large jobs
    
    n_small = n - n_large
    
    # Small jobs: 5-10 duration
    p_small = rng.integers(5, 11, size=n_small)
    # Large jobs: 30-50 duration
    p_large = rng.integers(30, 51, size=n_large)
    
    p = np.concatenate([p_small, p_large])
    rng.shuffle(p)  # Randomize order
    
    w = rng.integers(1, 4, size=n)

    df = pd.DataFrame(
        {
            "job": np.arange(1, n + 1),
            "p_i": p,
            "priority_w": w,
        }
    )

    df.to_csv(file_path, index=False)
    print(f"Generated clustered dataset: {file_path} (seed={seed}, {n_small} small, {n_large} large, mean={p.mean():.1f})")


def generate_dataset_by_type(
    n: int,
    file_path: str,
    instance_type: Literal["uniform", "heavy_tailed", "clustered", "default"],
    seed: Optional[int] = None,
):
    """Generate dataset based on instance type."""
    if instance_type == "uniform":
        generate_dataset_uniform(n, file_path, seed)
    elif instance_type == "heavy_tailed":
        generate_dataset_heavy_tailed(n, file_path, seed)
    elif instance_type == "clustered":
        generate_dataset_clustered(n, file_path, seed)
    else:  # default
        generate_dataset(n, file_path, seed)
