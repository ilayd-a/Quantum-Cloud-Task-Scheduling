import pandas as pd
import numpy as np

def generate_dataset(n, file_path):
    p = np.random.randint(5, 20, size=n)
    w = np.random.randint(1, 4, size=n)

    df = pd.DataFrame({
        "job": np.arange(1, n+1),
        "p_i": p,
        "priority_w": w
    })

    df.to_csv(file_path, index=False)
    print(f"Generated dataset: {file_path}")
