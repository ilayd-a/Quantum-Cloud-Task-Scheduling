import numpy as np
from typing import Sequence


def build_qubo(
    tasks: Sequence[dict],
    machines: int,
    assignment_penalty: float,
    balance_strength: float,
    priority_weight: float,
    machine_bias: Sequence[float] | None = None,
):
    """
    General M-machine scheduling QUBO using one-hot assignment variables.

    Variables: x_{i,m} = 1 if task i runs on machine m.
    Objective: enforce one-hot feasibility, penalise load imbalance and add
    a small linear priority bias favouring "better" machines.
    """

    if machines < 2:
        raise ValueError("machines must be >= 2 for the balanced scheduling QUBO.")

    proc = np.array([float(t["p_i"]) for t in tasks], dtype=float)
    weights = np.array([float(t.get("priority_w", 1.0)) for t in tasks], dtype=float)
    total_load = float(proc.sum())

    num_tasks = len(tasks)
    num_vars = num_tasks * machines
    Q = np.zeros((num_vars, num_vars))

    if machine_bias is None:
        # Lower bias value -> "better" machine. Default: machine 0 best.
        machine_bias = np.linspace(0.0, 1.0, machines)
    machine_bias = np.asarray(machine_bias, dtype=float)
    if machine_bias.shape[0] != machines:
        raise ValueError("machine_bias length must match the number of machines.")

    target_load = total_load / machines

    A = assignment_penalty
    B = balance_strength
    C = priority_weight

    def idx(i: int, m: int) -> int:
        return i * machines + m

    # 1) Assignment feasibility penalties: A * (1 - sum_m x_{i,m})^2
    for i in range(num_tasks):
        for m in range(machines):
            k = idx(i, m)
            Q[k, k] += -A  # linear coefficient (-A) * x_{i,m}
            for m2 in range(m + 1, machines):
                k2 = idx(i, m2)
                Q[k, k2] += 2 * A
                Q[k2, k] += 2 * A

    # 2) Load balancing penalties: B * sum_m (sum_i p_i x_{i,m} - target)^2
    for m in range(machines):
        for i in range(num_tasks):
            k = idx(i, m)
            Q[k, k] += B * (proc[i] ** 2) - 2 * B * target_load * proc[i]
            for j in range(i + 1, num_tasks):
                k2 = idx(j, m)
                coupling = 2 * B * proc[i] * proc[j]
                Q[k, k2] += coupling
                Q[k2, k] += coupling

    # 3) Priority bias (linear term only)
    for i in range(num_tasks):
        for m in range(machines):
            k = idx(i, m)
            Q[k, k] += C * weights[i] * proc[i] * machine_bias[m]

    return Q
