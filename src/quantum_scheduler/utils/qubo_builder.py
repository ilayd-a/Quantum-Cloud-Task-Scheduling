import numpy as np
from typing import Sequence


def build_qubo(
    tasks: Sequence[dict],
    imbalance_penalty: float,
    priority_weight: float,
    makespan_penalty: float | None = None,
):
    """
    Balanced 2-machine QUBO using a single binary variable per task.

    Binary variable x_i = 1 assigns task i to machine 1, otherwise to machine 0.
    Objective: A (2 L1 - T)^2 + C * L1^2 + B * sum_i w_i p_i x_i.
    
    The makespan_penalty term (C * L1^2) directly penalizes high loads on machine 1,
    which better aligns with minimizing makespan = max(L0, L1).
    """

    proc = np.array([float(t["p_i"]) for t in tasks], dtype=float)
    weights = np.array([float(t.get("priority_w", 1.0)) for t in tasks], dtype=float)

    total_load = float(proc.sum())
    A = float(imbalance_penalty)
    B = float(priority_weight)
    C = float(makespan_penalty) if makespan_penalty is not None else A * 2.0  # Default: 2x imbalance penalty

    n = len(proc)
    Q = np.zeros((n, n))

    for i in range(n):
        # Combined terms:
        # Balance: 2A p_i^2 - 4A total_load p_i  (from (L1 - L̄)^2)
        # Makespan: C p_i^2  (direct penalty on L1^2)
        # Priority: -B w_i p_i
        Q[i, i] = (2*A + C) * proc[i] ** 2 - 4 * A * total_load * proc[i] - B * weights[i] * proc[i]
        for j in range(i + 1, n):
            # Balance: 4A p_i p_j
            # Makespan: 2C p_i p_j
            coupling = (4*A + 2*C) * proc[i] * proc[j]
            Q[i, j] = coupling
            Q[j, i] = coupling  # Ensure symmetry

    return Q


def qubo_from_tasks(
    tasks: Sequence[dict],
    balance_penalty_multiplier: float | None = None,
    priority_bias: float = 0.1,
    makespan_penalty_multiplier: float | None = None,
):
    """
    Build a normalized QUBO matrix from raw task definitions.

    The processing times are normalized by the total load to keep the penalty scale
    consistent across datasets. The multiplier A is interpreted as
    balance_penalty_multiplier * total_load, but since the normalized load is 1.0,
    the actual penalty equals the multiplier.
    
    Args:
        tasks: List of task dictionaries
        balance_penalty_multiplier: Weight for load imbalance penalty
        priority_bias: Weight for priority term
        makespan_penalty_multiplier: Weight for direct makespan penalty (if None, uses 2x balance penalty)
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
    
    # If balance_penalty is 0, use a small default makespan penalty if not specified
    if makespan_penalty_multiplier is not None:
        makespan_mult = makespan_penalty_multiplier
    elif multiplier == 0.0:
        makespan_mult = 10.0  # Default when balance is disabled
    else:
        makespan_mult = multiplier * 2.0
    actual_makespan_penalty = makespan_mult

    Q = build_qubo(
        normalized_tasks,
        imbalance_penalty=actual_penalty,
        priority_weight=priority_bias,
        makespan_penalty=actual_makespan_penalty,
    )
    return Q, actual_penalty, multiplier, total_load
