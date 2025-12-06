import numpy as np
from typing import Sequence


def decode_solution_vector(x: Sequence[float], p: Sequence[int]):
    """
    Decode a binary assignment vector (length = number of tasks) into machine
    loads for the 2-machine case. x_i = 1 means machine 1, otherwise machine 0.
    """
    n = len(p)
    if len(x) != n:
        raise ValueError("Assignment vector length must match number of tasks.")

    x = np.asarray(x).round()
    loads = [0.0, 0.0]
    assignments = []

    for task_idx, machine_bit in enumerate(x):
        machine = int(machine_bit)
        assignments.append(machine)
        loads[machine] += p[task_idx]

    return {
        "assignment": assignments,
        "loads": loads,
        "makespan": max(loads),
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
