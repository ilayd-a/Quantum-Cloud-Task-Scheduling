# Quantum Cloud Task Scheduling

A research-grade sandbox for studying the two-machine load-balancing problem with both quantum approximate optimization (QAOA) and classical baselines. The codebase now follows a reproducible workflow: deterministic dataset generation, configurable QUBO construction, experiment orchestration via YAML, and automated table/figure synthesis for conference submissions.

## Repository Layout
```
├── configs/                  # YAML experiment sweeps (e.g., baseline.yaml)
├── data/datasets/            # CSV inputs regenerated via scripts
├── docs/report.md            # Manuscript-style project report
├── results/                  # JSON artifacts + CSV summaries (gitignored)
├── scripts/
│   ├── generate_datasets.py  # Deterministic synthetic workloads
│   ├── run_experiments.py    # Single runs or config-driven sweeps
│   └── analyze_results.py    # Markdown / LaTeX tables for publications
└── src/quantum_scheduler/    # Package with QUBO builder, solvers, utilities
```

## Quick Start
1. **Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -e .
   ```
2. **Datasets (deterministic seeds)**
   ```bash
   python scripts/generate_datasets.py --sizes 5 8 10 12 --seed 123
   ```
   This repopulates `data/datasets/` with CSVs keyed by job count.
3. **Single experiment**
   ```bash
   python scripts/run_experiments.py \
     --dataset dataset_10.csv --reps 2 --maxiter 80 \
     --balance-penalty 1.2 --priority-bias 0.4 --tag demo
   ```
   Produces `results/dataset_10_r2_shots4096_demo.json` containing QAOA energies, decoded schedules, and classical baselines.
4. **Batch sweep (recommended for papers)**
   ```bash
   python scripts/run_experiments.py --config configs/baseline.yaml
   ```
   Generates per-run JSON files and an aggregated CSV (`results/baseline_summary.csv`).
5. **Analysis & Tables**
   ```bash
   python scripts/analyze_results.py --input-csv results/baseline_summary.csv
   ```
   Emits `analysis/summary.md` + `analysis/summary.tex` ready for inclusion in the manuscript.

## Core Components
- `quantum_scheduler.utils.build_qubo`: Constructs a dense QUBO that penalizes load imbalance and rewards high-priority jobs on the designated primary machine.
- `quantum_scheduler.qaoa_solver.solve_qaoa_local`: Runs AerSimulator-based QAOA, optimizes parameters via COBYLA, samples the final circuit, and decodes the most promising bitstring into a two-machine schedule with makespan metrics.
- `quantum_scheduler.classical_solver.solve_classical`: Exact brute-force baseline (sufficient for ≤12 jobs) returning assignments, machine loads, and optimal makespans.
- `scripts/run_experiments.py`: CLI with support for config sweeps, per-run metadata, and relative performance gap computation.
- `scripts/analyze_results.py`: Converts JSON/CSV artifacts into publication-ready Markdown/LaTeX tables and prints summary statistics.

## Research Artefacts
- `docs/report.md` – Abstract, problem statement, methodology, and reproducibility checklist.
- `configs/baseline.yaml` – Default sweep referenced in the report.
- `analysis/summary.md|tex` – Auto-generated once analysis script is executed.

## Notes & Roadmap
- The current QUBO/decoder targets the two-machine partitioning variant; extending to `M>2` is a planned upgrade.
- PuLP is pinned to enable ILP baselines (scaffolding already exists in `quantum_scheduler/classical_solver.py`).
- For hardware validation, swap `AerSimulator` with provider-specific backends or noise models inside `solve_qaoa_local`.

With these additions, the repository satisfies the usual reproducibility requirements for quantum optimization workshops: sealed dependencies, deterministic data, declarative experiment configs, and scripted result synthesis.
# The General QUBO Form

$$
E(x) = x^T Q x + C^T x
$$

x: vector of binary variables $(x_1, x_2, \ldots, x_n)$  
→ in our case this is the flattened vector of $x_{i,m}$ job-machine assignments  

Q: symmetric matrix of quadratic coefficients  
→ captures pairwise interactions between binary variables  

c: vector of linear coefficients  
→ captures individual variable contributions  


---

# Problem

We want to assign a set of jobs:

$$
J = \{1, \ldots, n\}
$$

to a set of identical machines:

$$
M = \{1, \ldots, M\}
$$

so that:

1. every job goes to exactly one machine  
2. all machines have similar total workloads  
3. optionally, makespan is small  


---

# Decision Variables

$$
x_{i,m} =
\begin{cases}
1 & \text{if job } i \text{ runs on machine } m \\
0 & \text{otherwise}
\end{cases}
$$

Load on machine $m$:

$$
L_m = \sum_i p_i x_{i,m}
$$

Ideal average load:

$$
\bar{L} = \frac{1}{M} \sum_i p_i
$$


---

# Balanced Load Term

$$
E_{\text{balance}} = \sum_m (L_m - \bar{L})^2
$$

Substituting:

$$
E_{\text{balance}} =
\sum_m \left( \sum_i p_i x_{i,m} \right)^2
$$


---

# Assignment Constraint Term

$$
E_{\text{assign}} =
\sum_i \left( 1 - \sum_m x_{i,m} \right)^2
$$


---

# Full QUBO Objective

$$
E(x) =
\lambda_1 \sum_m \left( \sum_i p_i x_{i,m} \right)^2
+
\lambda_2 \sum_i \left( 1 - \sum_m x_{i,m} \right)^2
$$


Expanded generic form:

$$
E(x) =
\sum_{i,j,m,m'} Q_{(i,m),(j,m')} x_{i,m} x_{j,m'}
+
\sum_{i,m} c_{i,m} x_{i,m}
$$


---

# Q Matrix Structure

Diagonal (same job):

$$
Q_{(i,m),(i,m)} = -\lambda_2
$$

Off-diagonal (same job, different machines):

$$
Q_{(i,m),(i,m')} = 2\lambda_2
$$

Diagonal (same machine):

$$
Q_{(i,m),(i,m)} = \lambda_1 p_i^2
$$

Off-diagonal (same machine, different jobs):

$$
Q_{(i,m),(j,m)} = 2\lambda_1 p_i p_j
$$


---

# From QUBO to Ising Hamiltonian

$$
x_i \in \{0,1\} \rightarrow z_i \in \{-1,+1\}
$$

$$
x_i = \frac{1 - z_i}{2}
$$

Plug into $E(x)$:

$$
E(z) =
constant +
\sum_i h_i z_i +
\sum_{i<j} J_{i,j} z_i z_j
$$


---

# Circuit Structure

## 1. Cost Hamiltonian

$$
U_C(\gamma) = e^{-i \gamma H_C}
$$

## 2. Mixer Hamiltonian

$$
U_M(\beta) = e^{-i \beta H_M}
$$

$$
H_M = \sum_i X_i
$$


---

# Full QAOA Circuit

$$
|\psi(\gamma,\beta)\rangle =
U_M(\beta_p)
U_C(\gamma_p)
\cdots
U_M(\beta_1)
U_C(\gamma_1)
|+\rangle^{\otimes n}
$$


---

# Initial Superposition

$$
|\psi_0\rangle =
|+\rangle^{\otimes n}
=
\frac{1}{\sqrt{2^n}}
\sum_{x \in \{0,1\}^n} |x\rangle
$$


---

# QAOA State

$$
|\psi(\gamma,\beta)\rangle =
U_M(\beta)
U_C(\gamma)
|\psi_0\rangle
$$


---

# Probability of a Bitstring

$$
P(x) =
|\langle x | \psi(\gamma,\beta) \rangle|^2
$$


---

# Optimal Parameters

$$
(\gamma^*, \beta^*) =
\arg\min_{\gamma,\beta}
\langle
\psi(\gamma,\beta)
|
H_C
|
\psi(\gamma,\beta)
\rangle
$$
