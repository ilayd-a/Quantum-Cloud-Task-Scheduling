import numpy as np
from typing import Sequence


def decode_solution_vector(x: Sequence[float], p: Sequence[int], machines: int):
    """
    Decode a flattened assignment vector (length = n * machines) into an
    assignment list, per-machine loads, and makespan. Each job is assigned to
    the machine with the maximum indicator in its one-hot block.
    """
    n = len(p)
    x = np.asarray(x).reshape(n, machines)

    assignments = []
    loads = np.zeros(machines, dtype=float)

    for i in range(n):
        m = int(np.argmax(x[i]))
        assignments.append(m)
        loads[m] += p[i]

    return {
        "assignment": assignments,
        "loads": loads.tolist(),
        "makespan": float(loads.max()),
    }


def bitstring_to_vector(bitstring: str, num_qubits: int) -> np.ndarray:
    """
    Convert a measurement bitstring (Qiskit ordering) into a 0/1 numpy vector
    aligned with qubit indices (qubit 0 is the least significant bit).
    """
    cleaned = bitstring.replace(" ", "")
    if len(cleaned) < num_qubits:
        cleaned = cleaned.zfill(num_qubits)
    bits = cleaned[::-1]  # flip so index 0 corresponds to qubit 0
    return np.array([1 if b == "1" else 0 for b in bits[:num_qubits]], dtype=float)
