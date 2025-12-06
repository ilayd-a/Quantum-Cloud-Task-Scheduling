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
        # Negative sign rewards sending high-priority jobs to machine 1 (x_i=1 lowers energy).
        Q[i, i] = 4 * A * proc[i] ** 2 - 4 * A * total_load * proc[i] - B * weights[i] * proc[i]
        for j in range(i + 1, n):
            coupling = 8 * A * proc[i] * proc[j]
            Q[i, j] = coupling

    return Q


def qubo_from_tasks(
    tasks: Sequence[dict],
    balance_penalty_multiplier: float | None = None,
    priority_bias: float = 0.1,
):
    """
    Build a normalized QUBO matrix from raw task definitions.

    The processing times are normalized by the total load to keep the penalty scale
    consistent across datasets. The multiplier A is interpreted as
    balance_penalty_multiplier * total_load, but since the normalized load is 1.0,
    the actual penalty equals the multiplier.
    """

    proc = np.array([float(task.get("p_i", task.get("p"))) for task in tasks], dtype=float)
    total_load = float(proc.sum())
    if total_load <= 0:
        raise ValueError("Total processing time must be positive.")
    normalized_proc = proc / total_load

    normalized_tasks = []
    for task, p_norm in zip(tasks, normalized_proc):
        normalized_tasks.append(
            {
                "job": task.get("job"),
                "p_i": float(p_norm),
                "priority_w": task.get("priority_w", 1.0),
            }
        )

    multiplier = balance_penalty_multiplier if balance_penalty_multiplier is not None else 10.0
    actual_penalty = multiplier  # normalized total load is 1

    Q = build_qubo(
        normalized_tasks,
        imbalance_penalty=actual_penalty,
        priority_weight=priority_bias,
    )
    return Q, actual_penalty, multiplier, total_load
