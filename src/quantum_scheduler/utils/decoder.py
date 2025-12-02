import numpy as np

def decode_solution(x, p, M):
    n = len(p)
    x = np.array(x)

    assignments = []
    loads = np.zeros(M, dtype=int)

    for i in range(n):
        row = x[i*M:(i+1)*M]
        m = row.argmax()
        assignments.append(int(m))
        loads[m] += p[i]

    return assignments, loads.tolist(), int(loads.max())
