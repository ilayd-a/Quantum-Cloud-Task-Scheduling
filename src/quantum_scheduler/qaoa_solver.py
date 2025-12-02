import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit.circuit import Parameter
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import BackendSamplerV2


# ---------------------------------------------------------
# Convert simple tasks (job, p_i, w) → QUBO
# ---------------------------------------------------------
def qubo_from_tasks(tasks):
    """
    tasks = list of dicts: {"job":..., "p_i":..., "priority_w":...}
    QUBO objective = sum_i w_i * p_i * x_i
    """
    n = len(tasks)
    Q = np.zeros((n, n))

    for i, t in enumerate(tasks):
        p = float(t["p_i"])
        w = float(t["priority_w"])
        Q[i, i] = w * p

    return Q


# ---------------------------------------------------------
# Convert QUBO → Ising (h, J)
# ---------------------------------------------------------
def qubo_to_ising(Q):
    n = Q.shape[0]
    h = np.zeros(n)
    J = np.zeros((n, n))

    for i in range(n):
        h[i] = Q[i, i] / 2.0
        for j in range(i + 1, n):
            J[i, j] = Q[i, j] / 4.0
            J[j, i] = J[i, j]

    return h, J


# ---------------------------------------------------------
# Build QAOA circuit
# ---------------------------------------------------------
def qaoa_circuit(h, J, n, reps=1):
    qc = QuantumCircuit(n)

    beta_params = [Parameter(f"beta_{k}") for k in range(reps)]
    gamma_params = [Parameter(f"gamma_{k}") for k in range(reps)]

    # Initial Hadamards
    for i in range(n):
        qc.h(i)

    # QAOA layers
    for k in range(reps):
        beta = beta_params[k]
        gamma = gamma_params[k]

        # Cost Hamiltonian
        for i in range(n):
            qc.rz(2 * h[i] * gamma, i)

        for i in range(n):
            for j in range(i + 1, n):
                if abs(J[i, j]) > 1e-12:
                    qc.rzz(2 * J[i, j] * gamma, i, j)

        # Mixer
        for i in range(n):
            qc.rx(2 * beta, i)

    qc.measure_all()
    return qc, beta_params, gamma_params


# ---------------------------------------------------------
# Energy evaluation using SamplerV2
# ---------------------------------------------------------
def qaoa_energy(params, h, J, n, sampler, backend, qc, beta_params, gamma_params, shots=1024):

    bind_dict = {beta_params[0]: params[0], gamma_params[0]: params[1]}
    bound = qc.assign_parameters(bind_dict)

    result = sampler.run([bound], shots=shots).result()

    pub = result._pub_results[0]
    databin = pub.data

    bitarray = databin["meas"]

    # FINAL, CORRECT DECODER FOR YOUR QISKIT VERSION
    arr = bitarray.to_bool_array().astype(int)

    energy = 0.0
    for row in arr:
        z = np.where(row == 1, 1, -1)
        e = np.sum(h * z)
        for i in range(n):
            for j in range(i+1, n):
                e += J[i, j] * z[i] * z[j]
        energy += e

    return energy / len(arr)



# ---------------------------------------------------------
# Main local QAOA solver
# ---------------------------------------------------------
def solve_qaoa_local(tasks, reps=1, maxiter=30):
    Q = qubo_from_tasks(tasks)
    h, J = qubo_to_ising(Q)

    print(f"Converted QUBO → Ising. Num qubits: {len(tasks)}")

    n = len(tasks)
    qc, beta_params, gamma_params = qaoa_circuit(h, J, n, reps)

    # Local noisy Aer simulator
    noise_model = NoiseModel()
    backend = AerSimulator(noise_model=noise_model)

    sampler = BackendSamplerV2(backend=backend)

    def objective(par):
        return qaoa_energy(
            par, h, J, n,
            sampler,
            backend,
            qc, beta_params, gamma_params
        )

    optimizer = COBYLA(maxiter=maxiter)
    params0 = np.array([0.5, 0.5])

    print("Running classical optimization...")
    res = optimizer.minimize(objective, params0)

    return {
        "energy": res.fun,
        "optimal_params": res.x,
    }
