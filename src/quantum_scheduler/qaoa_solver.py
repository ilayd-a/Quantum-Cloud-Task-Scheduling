import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit.circuit import Parameter
from qiskit_algorithms.optimizers import COBYLA, SPSA

from .utils import qubo_from_tasks
from .utils.decoder import bitstring_to_vector, decode_solution_vector


def qubo_to_ising(Q):
    n = Q.shape[0]
    h = np.zeros(n)
    J = np.zeros((n, n))

    for i in range(n):
        h[i] = Q[i, i] / 2.0
        for j in range(n):
            if i != j:
                h[i] += Q[i, j] / 4.0
        for j in range(i + 1, n):
            J[i, j] = Q[i, j] / 4.0
            J[j, i] = J[i, j]

    return h, J



def qaoa_circuit(h, J, n, reps=1, measure=False):
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

    if measure:
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


def select_best_makespan_topk(counts: dict[str, int], tasks, K: int = 50):
    """
    Select the best schedule from top-K most probable bitstrings by makespan.
    
    Args:
        counts: Dictionary mapping bitstrings to measurement counts
        tasks: List of task dictionaries with 'p_i' processing times
        K: Number of top bitstrings to consider (default: 50)
    
    Returns:
        Tuple of (best_bitstring, best_schedule, best_makespan) or (None, None, None) if no valid samples
    """
    if not counts:
        return None, None, None
    
    p = [float(task["p_i"]) for task in tasks]
    num_qubits = len(tasks)
    
    # Sort bitstrings by count (most probable first)
    sorted_bitstrings = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
    # Take top-K
    top_k_bitstrings = sorted_bitstrings[:K]
    
    best_bitstring = None
    best_schedule = None
    best_makespan = float("inf")
    
    for bitstring, count in top_k_bitstrings:
        vector = bitstring_to_vector(bitstring, num_qubits)
        schedule = decode_solution_vector(vector, p)
        makespan = schedule["makespan"]
        
        if makespan < best_makespan:
            best_makespan = makespan
            best_bitstring = bitstring
            best_schedule = schedule
    
    if best_bitstring is None:
        return None, None, None
    
    return best_bitstring, best_schedule, best_makespan


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


def _prob_dict_expectation(prob_dict, h, J, n):
    energy = 0.0
    for bitstring, prob in prob_dict.items():
        energy += prob * _bitstring_energy(bitstring, h, J, n)
    return energy


def _prob_dict_to_counts(prob_dict, shots):
    counts = {}
    for bitstring, prob in prob_dict.items():
        counts[bitstring] = int(round(prob * shots))
    # ensure total counts == shots
    diff = shots - sum(counts.values())
    if diff > 0:
        for bitstring in sorted(prob_dict, key=prob_dict.get, reverse=True):
            counts[bitstring] += 1
            diff -= 1
            if diff == 0:
                break
    return counts


def _build_optimizer(name: str, maxiter: int):
    name = (name or "cobyla").lower()
    if name == "spsa":
        return SPSA(maxiter=maxiter)
    return COBYLA(maxiter=maxiter)


def qaoa_energy(params, h, J, n, eval_fn):
    return eval_fn(params, h, J, n)


def solve_qaoa_local(
    tasks,
    reps=1,
    maxiter=30,
    shots=1024,
    final_shots=4096,
    balance_penalty: float | None = None,
    priority_bias: float = 0.1,
    makespan_penalty: float | None = None,
    optimizer: str = "cobyla",
    restarts: int = 5,
    backend_type: str = "aer",
    seed: int | None = None,
    top_k: int = 50,
):
    """
    Optimize a QAOA circuit locally on AerSimulator and return both the energy
    landscape and the best sampled schedule decoded from measurement counts.
    """
    Q, actual_penalty, penalty_multiplier, total_load = qubo_from_tasks(
        tasks,
        balance_penalty_multiplier=balance_penalty,
        priority_bias=priority_bias,
        makespan_penalty_multiplier=makespan_penalty,
    )
    h, J = qubo_to_ising(Q)

    num_qubits = len(tasks)

    print(f"Converted QUBO → Ising. Num qubits: {num_qubits}")

    qc_base, beta_params, gamma_params = qaoa_circuit(h, J, num_qubits, reps, measure=False)

    backend_choice = (backend_type or "aer").lower()
    use_statevector = backend_choice == "statevector"
    if use_statevector:
        backend = None
    else:
        noise_model = NoiseModel()
        backend = AerSimulator(noise_model=noise_model)
        qc_meas = qc_base.copy()
        qc_meas.measure_all()

    num_params = len(beta_params) + len(gamma_params)

    rng = np.random.default_rng(seed)
    eval_counter = {"count": 0}
    energy_trace: list[dict[str, float | int]] = []
    current_restart = {"value": 1}

    def eval_fn(params, _, __, ___):
        eval_counter["count"] += 1
        bound = qc_base.assign_parameters(_assign_parameters(params, beta_params, gamma_params))
        if use_statevector:
            from qiskit.quantum_info import Statevector

            state = Statevector(bound)
            prob_dict = state.probabilities_dict()
            value = _prob_dict_expectation(prob_dict, h, J, num_qubits)
        else:
            measured = qc_meas.assign_parameters(_assign_parameters(params, beta_params, gamma_params))
            counts = _sample_counts(measured, backend, shots)
            value = _counts_expectation(counts, h, J, num_qubits)

        energy_trace.append(
            {
                "restart": current_restart["value"],
                "evaluation": eval_counter["count"],
                "energy": float(value),
            }
        )
        return value

    best_res = None
    best_value = float("inf")

    for r in range(max(restarts, 1)):
        current_restart["value"] = r + 1
        if num_params:
            params0 = rng.uniform(0, 2 * np.pi, size=num_params)
        else:
            params0 = np.array([])
        opt = _build_optimizer(optimizer, maxiter)

        print(f"Running optimisation restart {r+1}/{max(restarts,1)} with {optimizer.upper()}...")
        res = opt.minimize(lambda par: qaoa_energy(par, h, J, num_qubits, eval_fn), params0)

        if res.fun < best_value:
            best_value = res.fun
            best_res = res

    best_params = best_res.x if best_res is not None else params0
    bound = qc_base.assign_parameters(_assign_parameters(best_params, beta_params, gamma_params))

    if use_statevector:
        from qiskit.quantum_info import Statevector

        state = Statevector(bound)
        prob_dict = state.probabilities_dict()
        final_counts = _prob_dict_to_counts(prob_dict, final_shots)
    else:
        measured = qc_meas.assign_parameters(_assign_parameters(best_params, beta_params, gamma_params))
        final_counts = _sample_counts(measured, backend, final_shots)

    best_bitstring, best_sample_energy = _select_best_bitstring(final_counts, h, J, num_qubits)
    if best_sample_energy is not None:
        best_sample_energy = float(best_sample_energy)

    # Find the best makespan among all sampled bitstrings (for debugging/analysis)
    p = [float(task["p_i"]) for task in tasks]
    best_makespan_bitstring = None
    best_makespan_value = float("inf")
    min_makespan_in_samples = None
    
    for bitstring, count in final_counts.items():
        vector = bitstring_to_vector(bitstring, num_qubits)
        schedule_temp = decode_solution_vector(vector, p)
        makespan = schedule_temp["makespan"]
        if makespan < best_makespan_value:
            best_makespan_value = makespan
            best_makespan_bitstring = bitstring
        if min_makespan_in_samples is None or makespan < min_makespan_in_samples:
            min_makespan_in_samples = makespan

    schedule = None
    if best_bitstring is not None:
        vector = bitstring_to_vector(best_bitstring, num_qubits)
        schedule = decode_solution_vector(vector, p)

    # Top-K postselection by makespan
    topk_bitstring, topk_schedule, topk_makespan = select_best_makespan_topk(
        final_counts, tasks, K=top_k
    )

    energy_sorted = sorted(
        final_counts.keys(),
        key=lambda bit: _bitstring_energy(bit, h, J, num_qubits),
    )
    top_energy_solutions = []
    seen = set()
    for bit in energy_sorted:
        if bit in seen:
            continue
        seen.add(bit)
        vector = bitstring_to_vector(bit, num_qubits)
        schedule_info = decode_solution_vector(vector, [float(task["p_i"]) for task in tasks])
        top_energy_solutions.append(
            {
                "bitstring": bit,
                "energy": _bitstring_energy(bit, h, J, num_qubits),
                "makespan": schedule_info["makespan"],
                "loads": schedule_info["loads"],
            }
        )
        if len(top_energy_solutions) == 3:
            break

    return {
        "energy": float(best_value),
        "optimal_params": best_params.tolist(),
        "best_bitstring": best_bitstring,
        "best_sample_energy": best_sample_energy,
        "best_makespan_bitstring": best_makespan_bitstring,
        "min_makespan_in_samples": float(min_makespan_in_samples) if min_makespan_in_samples is not None else None,
        "counts": {k: int(v) for k, v in final_counts.items()},
        "best_schedule": schedule,  # Min-energy selection schedule
        "topk_bitstring": topk_bitstring,
        "topk_schedule": topk_schedule,  # Top-K postselection schedule
        "topk_makespan": float(topk_makespan) if topk_makespan is not None else None,
        "top_k": top_k,
        "shots": shots,
        "final_shots": final_shots,
        "balance_penalty_multiplier": penalty_multiplier,
        "balance_penalty_actual": actual_penalty,
        "total_load": total_load,
        "priority_bias": priority_bias,
        "optimizer": optimizer,
        "restarts": restarts,
        "backend": backend_choice,
        "evaluations": eval_counter["count"],
        "energy_trace": energy_trace,
        "top_solutions": top_energy_solutions,
    }
