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
        p = t.get("p", t.get("p_i"))
        w = t.get("w", t.get("priority_w"))

        if p is None or w is None:
            raise KeyError(
                "Each task must define either (p, w) or (p_i, priority_w)."
            )

        Q[i, i] = float(p) * float(w)

    return Q
