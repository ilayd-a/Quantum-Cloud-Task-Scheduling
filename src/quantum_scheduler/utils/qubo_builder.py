import numpy as np
from typing import Sequence


def build_qubo(
    tasks: Sequence[dict],
    imbalance_penalty: float,
    priority_weight: float,
):
    """
    Balanced 2-machine QUBO using a single binary variable per task.

    Binary variable x_i = 1 assigns task i to machine 1, otherwise to machine 0.
    Objective: A (2 L1 - T)^2 + B * sum_i w_i p_i x_i.
    """

    proc = np.array([float(t["p_i"]) for t in tasks], dtype=float)
    weights = np.array([float(t.get("priority_w", 1.0)) for t in tasks], dtype=float)

    total_load = float(proc.sum())
    A = float(imbalance_penalty)
    B = float(priority_weight)

    n = len(proc)
    Q = np.zeros((n, n))

    for i in range(n):
        Q[i, i] = 4 * A * proc[i] ** 2 - 4 * A * total_load * proc[i] + B * weights[i] * proc[i]
        for j in range(i + 1, n):
            coupling = 8 * A * proc[i] * proc[j]
            Q[i, j] = coupling

    return Q
