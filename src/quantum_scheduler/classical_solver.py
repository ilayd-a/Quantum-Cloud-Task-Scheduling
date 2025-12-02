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
