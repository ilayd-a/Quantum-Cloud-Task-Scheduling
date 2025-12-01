# utils/qubo_builder.py
import numpy as np

def build_qubo(tasks):
    """
    Builds a simple QUBO: minimize weighted processing time.
    Q[i, i] = w_i * p_i
    No cross terms for now (keeps QAOA fast and clean)
    """
    n = len(tasks)
    Q = np.zeros((n, n))

    for i, t in enumerate(tasks):
        Q[i, i] = t["p"] * t["w"]

    return Q
