import numpy as np
from typing import Sequence


def _get_processing_and_priority(tasks: Sequence[dict]):
    proc = []
    priority = []
    for task in tasks:
        p = task.get("p", task.get("p_i"))
        w = task.get("w", task.get("priority_w", 1))

        if p is None:
            raise KeyError("Task entry missing processing time (p or p_i).")

        proc.append(float(p))
        priority.append(float(w))
    return np.array(proc, dtype=float), np.array(priority, dtype=float)


def build_qubo(
    tasks: Sequence[dict],
    balance_penalty: float = 1.0,
    priority_bias: float = 0.25,
):
    """
    Build a QUBO that encourages balanced partitions between two machines while
    still preferring high-priority tasks to be executed early.

    The binary variable x_i = 1 indicates task i is sent to machine 1, whereas
    x_i = 0 keeps it on machine 0. We minimize the squared load difference
    between the two machines and add a lightweight linear bias proportional to
    task priority.
    """

    proc, priority = _get_processing_and_priority(tasks)
    n = len(proc)
    total_load = proc.sum()

    Q = np.zeros((n, n))

    for i in range(n):
        diag = balance_penalty * (4 * proc[i] * (proc[i] - total_load))
        diag += priority_bias * priority[i] * proc[i]
        Q[i, i] += diag

        for j in range(i + 1, n):
            coupling = balance_penalty * (8 * proc[i] * proc[j])
            Q[i, j] += coupling
            Q[j, i] += coupling

    return Q
