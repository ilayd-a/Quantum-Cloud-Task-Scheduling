from itertools import product
from typing import Iterable, Sequence


def solve_classical(p: Sequence[int], M: int = 3):
    """Brute-force classical baseline for the M-machine scheduling problem."""
    n = len(p)
    best_makespan = float("inf")
    best_assignment = None
    best_loads: Iterable[int] | None = None

    for assignment in product(range(M), repeat=n):
        loads = [0] * M
        for i, m in enumerate(assignment):
            loads[m] += p[i]

        makespan = max(loads)

        if makespan < best_makespan:
            best_makespan = makespan
            best_assignment = assignment
            best_loads = loads.copy()

    if best_loads is None:
        best_loads = [0] * M

    return {
        "assignment": best_assignment,
        "loads": [int(l) for l in best_loads],
        "makespan": int(best_makespan),
    }


def solve_greedy(p: Sequence[int], M: int = 2):
    """
    Simple greedy heuristic: assign each task to the machine with the lowest current load.
    """
    loads = [0] * M
    assignment = []

    for duration in p:
        machine = min(range(M), key=lambda m: loads[m])
        loads[machine] += duration
        assignment.append(machine)

    return {
        "assignment": assignment,
        "loads": loads,
        "makespan": max(loads),
    }
