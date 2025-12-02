import numpy as np
from typing import Iterable, Sequence


def decode_solution(x: Sequence[float], p: Sequence[int], M: int):
    """Original decoder for flattened assignment vectors of length n * M."""
    n = len(p)
    x = np.array(x)

    assignments = []
    loads = np.zeros(M, dtype=int)

    for i in range(n):
        row = x[i * M : (i + 1) * M]
        m = row.argmax()
        assignments.append(int(m))
        loads[m] += p[i]

    return assignments, loads.tolist(), int(loads.max())


def decode_binary_assignment(bitstring: str, tasks: Sequence[dict], machines: int = 2):
    """
    Interpret a single register bitstring (produced by the QAOA circuit) as a
    binary assignment for a two-machine partitioning problem.

    We adopt Qiskit's little-endian convention: the rightmost character
    corresponds to qubit 0 / task 0. A '1' indicates the task is assigned to
    machine 1, while '0' keeps it on machine 0. For now this decoder only
    supports the two-machine case used in our QUBO construction.
    """
    if machines != 2:
        raise ValueError(
            "decode_binary_assignment currently supports exactly 2 machines."
        )

    clean_bits = bitstring.replace(" ", "")[::-1]
    n = len(tasks)
    if len(clean_bits) < n:
        clean_bits = clean_bits.zfill(n)

    loads = [0] * machines
    assignment = []

    for idx in range(n):
        task = tasks[idx]
        processing = int(task.get("p_i", task.get("p", 0)))
        bit = clean_bits[idx]
        machine = 1 if bit == "1" else 0
        assignment.append(machine)
        loads[machine] += processing

    return {
        "assignment": assignment,
        "loads": loads,
        "makespan": max(loads) if loads else 0,
        "bitstring": bitstring,
    }
