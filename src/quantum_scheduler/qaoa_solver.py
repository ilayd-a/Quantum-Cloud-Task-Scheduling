import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit.circuit import Parameter
from qiskit_algorithms.optimizers import COBYLA

from .utils import build_qubo
from .utils.decoder import bitstring_to_vector, decode_solution_vector


def _processing_times(tasks):
    return np.array([float(task.get("p_i", task.get("p"))) for task in tasks], dtype=float)


def qubo_from_tasks(
    tasks,
    balance_penalty_multiplier: float | None = None,
    priority_bias: float = 0.1,
):
    """
    Build the balanced-load QUBO and report the actual penalty strength used.

    The multiplier is interpreted as A = multiplier * total_load, defaulting to 10.
    """
    proc = _processing_times(tasks)
    total_load = float(proc.sum())
    multiplier = balance_penalty_multiplier if balance_penalty_multiplier is not None else 10.0
    actual_penalty = multiplier * total_load

    Q = build_qubo(
        tasks,
        imbalance_penalty=actual_penalty,
        priority_weight=priority_bias,
    )
    return Q, actual_penalty, multiplier, total_load


def qubo_to_ising(Q):
    """Convert a QUBO matrix Q into Ising parameters (h, J)."""
    n = Q.shape[0]
    h = np.zeros(n)
    J = np.zeros((n, n))

    for i in range(n):
        h[i] = Q[i, i] / 2.0
        for j in range(i + 1, n):
            J[i, j] = Q[i, j] / 4.0
            J[j, i] = J[i, j]

    return h, J


def qaoa_circuit(h, J, n, reps=1):
    """Construct the parameterized QAOA circuit."""
    qc = QuantumCircuit(n)

    beta_params = [Parameter(f"beta_{k}") for k in range(reps)]
    gamma_params = [Parameter(f"gamma_{k}") for k in range(reps)]

    for i in range(n):
        qc.h(i)

    for k in range(reps):
        beta = beta_params[k]
        gamma = gamma_params[k]

        for i in range(n):
            qc.rz(2 * h[i] * gamma, i)

        for i in range(n):
            for j in range(i + 1, n):
                if abs(J[i, j]) > 1e-12:
                    qc.rzz(2 * J[i, j] * gamma, i, j)

        for i in range(n):
            qc.rx(2 * beta, i)

    qc.measure_all()
    return qc, beta_params, gamma_params


def _bitstring_to_z(bitstring: str, n: int) -> np.ndarray:
    cleaned = bitstring.replace(" ", "")
    bits = cleaned.zfill(n)[::-1]
    return np.array([1 if b == "1" else -1 for b in bits], dtype=float)


def _bitstring_energy(bitstring: str, h, J, n) -> float:
    z = _bitstring_to_z(bitstring, n)
    energy = float(np.dot(h, z))
    for i in range(n):
        for j in range(i + 1, n):
            energy += J[i, j] * z[i] * z[j]
    return energy


def _counts_expectation(counts: dict[str, int], h, J, n) -> float:
    total = sum(counts.values())
    if total == 0:
        raise ValueError("Backend returned zero shots; cannot compute energy.")

    exp_val = 0.0
    for bitstring, freq in counts.items():
        e = _bitstring_energy(bitstring, h, J, n)
        exp_val += e * (freq / total)
    return exp_val


def _select_best_bitstring(counts: dict[str, int], h, J, n):
    best_bitstring = None
    best_energy = float("inf")

    for bitstring in counts:
        e = _bitstring_energy(bitstring, h, J, n)
        if e < best_energy:
            best_energy = e
            best_bitstring = bitstring

    return best_bitstring, best_energy


def _assign_parameters(params, beta_params, gamma_params):
    if len(params) != len(beta_params) + len(gamma_params):
        raise ValueError(
            "Parameter vector length does not match the requested number of QAOA layers."
        )

    bind_dict = {}
    offset = 0
    for beta in beta_params:
        bind_dict[beta] = params[offset]
        offset += 1
    for gamma in gamma_params:
        bind_dict[gamma] = params[offset]
        offset += 1
    return bind_dict


def _sample_counts(bound_circuit, backend, shots):
    result = backend.run(bound_circuit, shots=shots).result()
    counts = result.get_counts()
    if isinstance(counts, list):
        counts = counts[0]
    return counts


def qaoa_energy(params, h, J, n, backend, qc, beta_params, gamma_params, shots=1024):
    bound = qc.assign_parameters(_assign_parameters(params, beta_params, gamma_params))
    counts = _sample_counts(bound, backend, shots)
    return _counts_expectation(counts, h, J, n)


def solve_qaoa_local(
    tasks,
    reps=1,
    maxiter=30,
    shots=1024,
    final_shots=4096,
    balance_penalty: float | None = None,
    priority_bias: float = 0.1,
):
    """
    Optimize a QAOA circuit locally on AerSimulator and return both the energy
    landscape and the best sampled schedule decoded from measurement counts.
    """
    Q, actual_penalty, penalty_multiplier, total_load = qubo_from_tasks(
        tasks,
        balance_penalty_multiplier=balance_penalty,
        priority_bias=priority_bias,
    )
    h, J = qubo_to_ising(Q)

    num_qubits = len(tasks)

    print(f"Converted QUBO → Ising. Num qubits: {num_qubits}")

    qc, beta_params, gamma_params = qaoa_circuit(h, J, num_qubits, reps)

    noise_model = NoiseModel()
    backend = AerSimulator(noise_model=noise_model)

    def objective(par):
        return qaoa_energy(
            par,
            h,
            J,
            num_qubits,
            backend,
            qc,
            beta_params,
            gamma_params,
            shots=shots,
        )

    optimizer = COBYLA(maxiter=maxiter)
    num_params = len(beta_params) + len(gamma_params)
    params0 = np.full(num_params, 0.5)

    print("Running classical optimization...")
    res = optimizer.minimize(objective, params0)

    best_params = res.x
    bound = qc.assign_parameters(_assign_parameters(best_params, beta_params, gamma_params))
    final_counts = _sample_counts(bound, backend, final_shots)
    best_bitstring, best_sample_energy = _select_best_bitstring(final_counts, h, J, num_qubits)
    if best_sample_energy is not None:
        best_sample_energy = float(best_sample_energy)

    schedule = None
    if best_bitstring is not None:
        vector = bitstring_to_vector(best_bitstring, num_qubits)
        p = [float(task["p_i"]) for task in tasks]
        schedule = decode_solution_vector(vector, p)

    return {
        "energy": float(res.fun),
        "optimal_params": best_params.tolist(),
        "best_bitstring": best_bitstring,
        "best_sample_energy": best_sample_energy,
        "counts": {k: int(v) for k, v in final_counts.items()},
        "best_schedule": schedule,
        "shots": shots,
        "final_shots": final_shots,
        "balance_penalty_multiplier": penalty_multiplier,
        "balance_penalty_actual": actual_penalty,
        "total_load": total_load,
        "priority_bias": priority_bias,
    }
