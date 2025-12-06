import numpy as np
import pandas as pd
from typing import Optional


def generate_dataset(n: int, file_path: str, seed: Optional[int] = None):
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
