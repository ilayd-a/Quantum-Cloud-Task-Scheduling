import numpy as np
from typing import Sequence


def build_qubo(
    tasks: Sequence[dict],
    penalty: float | None = None,
    priority_weight: float = 1.0,
):
    """
    Balanced 2-machine scheduling QUBO.
    Minimizes load imbalance and applies a lightweight priority bias.
    """

    p = np.array([float(t["p_i"]) for t in tasks])
    w = np.array([float(t.get("priority_w", 1.0)) for t in tasks])
    total_load = p.sum()

    # Literature-friendly default: A = 10 * total_load
    A = penalty if penalty is not None else 10.0 * total_load
    B = priority_weight

    n = len(p)
    Q = np.zeros((n, n))

    for i in range(n):
        Q[i, i] = (
            4 * A * p[i] ** 2
            - 4 * A * total_load * p[i]
            + B * w[i] * p[i]
        )

        for j in range(i + 1, n):
            # Coupling derived from (sum_i p_i (2x_i - 1))^2 expansion
            coupling = 8 * A * p[i] * p[j]
            Q[i, j] = Q[j, i] = coupling

    return Q
