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

**QUBO Objective:**  
$E(x) = x^T Q x + C^T x$

- **x:** vector of binary variables $(x_1, x_2, \ldots, x_n)$  
  → In our case, this is the flattened vector of $x_{i,m}$ job–machine assignments.  
- **Q:** symmetric matrix of quadratic coefficients  
  → Captures pairwise interactions between binary variables.  
- **c:** vector of linear coefficients  
  → Captures individual variable contributions (cost or bias).  

---

# Problem Setup

We have:

**Jobs:**  
$J = \{1, \ldots, n\}$  

**Machines:**  
$M = \{1, \ldots, M\}$  

Goals:
1. Each job is assigned to exactly one machine.  
2. All machines have similar workloads.  
3. Optionally minimize makespan.

---

# Decision Variables

**Binary assignment variable:**  
$x_{i,m} = 1$ if job $i$ runs on machine $m$, else $0$.

**Machine load:**  
$L_m = \sum_i p_i x_{i,m}$  

**Average load:**  
$\bar{L} = \frac{1}{M} \sum_i p_i$

---

# Balanced Load Penalty

**Balance term:**  
$E_{\text{balance}} = \sum_m (L_m - \bar{L})^2$

After substituting $L_m = \sum_i p_i x_{i,m}$:

**Expanded balance term:**  
$E_{\text{balance}} = \sum_m \left( \sum_i p_i x_{i,m} \right)^2$

Scaled by $\lambda_1$.

---

# Assignment Constraint Penalty

**One-hot assignment condition:**  
$\sum_m x_{i,m} = 1$

Convert to penalty:

**Assignment penalty:**  
$E_{\text{assign}} = \sum_i \left( 1 - \sum_m x_{i,m} \right)^2$

Scaled by $\lambda_2$.

---

# Full QUBO Energy

**Total energy:**  
$E(x) = \lambda_1 \sum_m (\sum_i p_i x_{i,m})^2 + \lambda_2 \sum_i (1 - \sum_m x_{i,m})^2$

Generic quadratic form:

**General QUBO:**  
$E(x) = x^T Q x + c^T x + \text{constant}$

---

# Q-Matrix Structure

## Machine-level interactions (from $\lambda_1$)

**Diagonal:**  
$Q_{(i,m),(i,m)} = \lambda_1 p_i^2$

**Off-diagonal (same machine, different jobs):**  
$Q_{(i,m),(j,m)} = 2 \lambda_1 p_i p_j$

---

## Job-level interactions (from $\lambda_2$)

**Diagonal:**  
$Q_{(i,m),(i,m)} = -\lambda_2$

**Off-diagonal (same job, different machines):**  
$Q_{(i,m),(i,m')} = 2 \lambda_2$

---

# From QUBO to Ising Hamiltonian

**Binary–spin conversion:**  
$x_i \in \{0,1\} \rightarrow z_i \in \{-1,+1\}$  

**Mapping:**  
$x_i = \frac{1 - z_i}{2}$

**Ising form:**  
$E(z) = \text{constant} + \sum_i h_i z_i + \sum_{i<j} J_{i,j} z_i z_j$

Where:
- $h_i$ are linear coefficients  
- $J_{i,j}$ are coupling strengths  

---

# Circuit Structure (QAOA)

## 1. Cost Hamiltonian Layer

**Cost unitary:**  
$U_C(\gamma) = e^{-i \gamma H_C}$

## 2. Mixer Hamiltonian Layer

**Mixer unitary:**  
$U_M(\beta) = e^{-i \beta H_M}$

**Mixer Hamiltonian:**  
$H_M = \sum_i X_i$

---

# Full QAOA Circuit

**Circuit of depth $p$:**  
$|\psi(\gamma,\beta)\rangle =  
U_M(\beta_p) U_C(\gamma_p) \cdots U_M(\beta_1) U_C(\gamma_1) |+\rangle^{\otimes n}$

Qubits represent binary decision variables $(x_{i,m})$.

---

# Initial State

**Initial superposition:**  
$|\psi_0\rangle = |+\rangle^{\otimes n} = \frac{1}{\sqrt{2^n}} \sum_{x \in \{0,1\}^n} |x\rangle$

---

# QAOA State

**State after parameters $(\gamma,\beta)$:**  
$|\psi(\gamma,\beta)\rangle = U_M(\beta) U_C(\gamma) |\psi_0\rangle$

---

# Measurement Probability

**Probability of observing bitstring $x$:**  
$P(x) = |\langle x | \psi(\gamma,\beta)\rangle|^2$

---

# Optimal Parameters

**Optimal angles:**  
$(\gamma^*, \beta^*) = \arg\min_{\gamma,\beta} \langle \psi(\gamma,\beta) | H_C | \psi(\gamma,\beta) \rangle$
