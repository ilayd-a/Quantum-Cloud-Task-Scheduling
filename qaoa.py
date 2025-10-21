import numpy as np
from itertools import product

from qiskit_aer.primitives import Sampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
import pandas as pd
from qiskit_algorithms.utils import algorithm_globals

# small index helper to flatten (i,m)-> a single index
def idx(i, m, M):
    return i*M + m

# read data
data = pd.read_csv("syntheticDataSet.csv")

p = data["p_i"].to_numpy()          # processing times
w = data["priority_w"].to_numpy()   # weights/priorities
n = len(p)
M = 3   # number of machines

lam_balance = 1.0
lam_assign = 5.0
num_vars = n * M
Q = np.zeros((num_vars, num_vars))
c = np.zeros(num_vars)
const = 0.0
effective_p = p * w

def idx(i, m):
    return i * M + m

# (a) balance term(lambda 1)
for m in range(M):
    for i in range(n):
        Q[idx(i,m), idx(i,m)] += lam_balance * (effective_p[i]**2)
    for i in range(n):
        for j in range(i+1, n):
            ii, jj = idx(i,m), idx(j,m)
            coef = 2 * lam_balance * effective_p[i]*effective_p[j]
            Q[ii, jj] += coef
            Q[jj, ii] += coef

# (b) assignment constraint term(lambda 2)
for i in range(n):
    const += lam_assign
    for m in range(M):
        Q[idx(i,m), idx(i,m)] += -lam_assign
    for m1 in range(M):
        for m2 in range(m1+1, M):
            ii, jj = idx(i,m1), idx(i,m2)
            Q[ii, jj] += 2*lam_assign
            Q[jj, ii] += 2*lam_assign

qp = QuadraticProgram()
for k in range(num_vars):
    qp.binary_var(name=f"x_{k}")

qp.minimize(quadratic=Q, linear=c, constant=const)
ising_op, offset = qp.to_ising()

algorithm_globals.random_seed = 42

sampler = Sampler()
qaoa = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=100), reps=2)
solver = MinimumEigenOptimizer(qaoa)
result = solver.solve(qp)
print(result)

counts = result.samples[0].data  # list of (bitstring, prob, energy)
print("Top samples:")
for s in result.samples[:5]:
    print(s)