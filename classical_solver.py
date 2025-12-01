import numpy as np
from itertools import product

from utils.decoder import decode_solution

def solve_classical(p, M=3):
    n = len(p)
    best_makespan = float("inf")
    best_assignment = None

    for assignment in product(range(M), repeat=n):
        loads = [0] * M
        for i, m in enumerate(assignment):
            loads[m] += p[i]

        makespan = max(loads)

        if makespan < best_makespan:
            best_makespan = makespan
            best_assignment = assignment

    return {
        "assignment": best_assignment,
        "loads": [int(l) for l in loads],
        "makespan": int(best_makespan)
    }
