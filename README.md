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
│   ├── analyze_results.py    # Markdown / LaTeX tables for publications
│   └── prove_advantage.py    # Turnkey script to showcase QAOA benefits
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

6. **Demonstrate QAOA advantages**
   ```bash
   python scripts/prove_advantage.py
   ```
   Runs three preconfigured experiments (quality trap, search-efficiency, diversity) that contrast QAOA with greedy and brute-force classical methods.

7. **Generate diagnostic plots**
   ```bash
   python scripts/plot_results.py --results-dir results --plots-dir results/plots
   ```
   Produces a full visualization suite per dataset:
   - QUBO landscape line plot (energy spread of sampled bitstrings)
   - Optimizer convergence curves (energy vs. evaluation)
   - Makespan comparison bar chart (QAOA vs. classical baselines)
   - Sample distribution histogram (top measured bitstrings)
   - Heatmap of QUBO coefficients (structure of the quadratic form)

8. **Render the QAOA circuit diagram**
   ```bash
   python scripts/render_circuit_diagram.py --dataset dataset_10.csv --reps 2 --measure
   ```
   Saves a publication-ready circuit figure to `results/circuit_diagrams/dataset_10_p2.png` (use `--drawer text` for ASCII output).

9. **Run the tests**
   ```bash
   pytest
   ```
   Exercises the plotting helpers to ensure regressions are caught early.

## Core Components
- `quantum_scheduler.utils.build_qubo`: Constructs a dense QUBO that penalizes load imbalance and rewards high-priority jobs on the designated primary machine.
- `quantum_scheduler.qaoa_solver.solve_qaoa_local`: Runs AerSimulator-based QAOA, optimizes parameters via COBYLA, samples the final circuit, and decodes the most promising bitstring into a two-machine schedule with makespan metrics.
- `quantum_scheduler.classical_solver.solve_classical`: Exact brute-force baseline (sufficient for ≤12 jobs) returning assignments, machine loads, and optimal makespans.
- `quantum_scheduler.classical_solver.solve_greedy`: Lightweight heuristic assigning each task to the currently lightest machine (used to showcase QAOA’s quality advantage).
- `scripts/run_experiments.py`: CLI with support for config sweeps, per-run metadata, and relative performance gap computation.
- `scripts/analyze_results.py`: Converts JSON/CSV artifacts into publication-ready Markdown/LaTeX tables and prints summary statistics.
- `scripts/prove_advantage.py`: Reproducible storytelling harness that runs the quality/efficiency/diversity tests contrasting QAOA with greedy/brute-force methods.

## Research Artefacts
- `docs/report.md` – Abstract, problem statement, methodology, and reproducibility checklist.
- `configs/baseline.yaml` – Default sweep referenced in the report.
- `analysis/summary.md|tex` – Auto-generated once analysis script is executed.

## Notes & Roadmap
- The current QUBO/decoder targets the two-machine partitioning variant; extending to `M>2` is a planned upgrade.
- PuLP is pinned to enable ILP baselines (scaffolding already exists in `quantum_scheduler/classical_solver.py`).
- For hardware validation, swap `AerSimulator` with provider-specific backends or noise models inside `solve_qaoa_local`.

With these additions, the repository satisfies the usual reproducibility requirements for quantum optimization workshops: sealed dependencies, deterministic data, declarative experiment configs, and scripted result synthesis.
## 1. The General QUBO Form

$$
E(x) = x^T Q x + c^T x
$$

x: vector of **binary variables** \((x_1, x_2, \ldots, x_n)\)  
- in our case this is the flattened vector of \(x_{i,m}\) job–machine assignments

Q: symmetric **matrix of quadratic coefficients**  
- captures pairwise interactions between binary variables  
- e.g., \(Q_{i,j} x_i x_j\) could mean “if two jobs are on the same machine, add a penalty”

c: vector of **linear coefficients**  
- captures individual variable contributions (cost or bias for each assignment)

E(x): total **energy / cost** to minimize

---

## 2. Problem Setup

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
3. optionally, total finish time (makespan) is small

---

## 3. Decision Variables

We create:

$$
x_{i,m} =
\begin{cases}
1 & \text{if job } i \text{ runs on machine } m,\\
0 & \text{otherwise}
\end{cases}
$$

These are the only unknowns the solver will decide.

If each job \(i\) has processing time \(p_i\),  
the load on machine \(m\) is:

$$
L_m = \sum_i p_i x_{i,m}
$$

The average load is:

$$
\bar{L} = \frac{1}{M} \sum_i p_i
$$

---

## 4. Balanced Load Penalty

Minimize how far each load is from the average:

$$
E_{\text{balance}} = \sum_m (L_m - \bar{L})^2 \approx \sum_m L_m^2
$$

- since \(\bar{L}\) is constant, subtracting it only adds a constant shift

Substituting \(L_m = \sum_i p_i x_{i,m}\):

$$
E_{\text{balance}} = \sum_m \left( \sum_i p_i x_{i,m} \right)^2
$$

- this is our first quadratic term  
- we scale it with weight \(\lambda_1\)

---

## 5. Assignment Constraint Penalty

Each job must go to exactly one machine:

$$
\sum_m x_{i,m} = 1
$$

Convert to a quadratic penalty:

$$
E_{\text{assign}} =
\sum_i \left( 1 - \sum_m x_{i,m} \right)^2
$$

- this enforces one-hot assignment  
- scaled by \(\lambda_2\)

Total QUBO objective:

$$
E(x) =
\lambda_1 \sum_m \left( \sum_i p_i x_{i,m} \right)^2
+
\lambda_2 \sum_i \left( 1 - \sum_m x_{i,m} \right)^2
$$

Quadratic form:

$$
E(x) = x^T Q x + c^T x + \text{constant}
$$

---

## 6. Building the Q Matrix + Ising Conversion

General expanded Q form:

$$
E(x)
=
\sum_{i,j,m,m'}
Q_{(i,m),(j,m')} \, x_{i,m} x_{j,m'}
+
\sum_{i,m} c_{i,m} x_{i,m}
+
\text{const}
$$

Interpretation:
- each pair \((i,m)\) is one binary variable  
- \(Q\) encodes interactions  
- \(c\) encodes linear preferences  

### a) Machine-level interactions (from \(\lambda_1\))

Diagonal:

$$
Q_{(i,m),(i,m)} = \lambda_1 p_i^2
$$

Because the variables are binary (\(x^2 = x\)), these diagonal entries ultimately act as the linear coefficients \(c_{i,m} = \lambda_1 p_i^2\) in \(c^T x\), even though we place them on the diagonal of \(Q\) for compactness.

Off-diagonal (within same machine):

$$
Q_{(i,m),(j,m)} = 2 \lambda_1 p_i p_j
$$

- this penalizes putting too many long jobs on the same machine

### b) Job-level interactions (from \(\lambda_2\))

Diagonal:

$$
Q_{(i,m),(i,m)} = -\lambda_2
$$

Again, due to \(x_{i,m}^2 = x_{i,m}\), this diagonal term corresponds to the linear bias \(c_{i,m} = -\lambda_2\).

Off-diagonal (within same job):

$$
Q_{(i,m),(i,m')} = 2 \lambda_2
$$

The full Q matrix is the sum of these effects (with the understanding that diagonal entries double as the linear coefficient vector).

---

### Converting QUBO to Ising

Binary → spin:

$$
x_i \in \{0,1\} \longrightarrow z_i \in \{-1,+1\}
$$

Mapping:

$$
x_i = \frac{1 - z_i}{2}
$$

Plugging into the energy yields the Ising Hamiltonian:

$$
E(z)
=
\text{constant}
+
\sum_i h_i z_i
+
\sum_{i<j} J_{i,j} z_i z_j
$$

- this is the Ising Hamiltonian form  
- linear coefficients \(h_i\) come from diagonal parts of Q  
- coupling terms \(J_{i,j}\) come from off-diagonal Q entries

---

## 7. QAOA Circuit Structure

QAOA alternates between Cost and Mixer unitaries.

### Cost Hamiltonian layer

$$
U_C(\gamma) = e^{-i \gamma H_C}
$$

### Mixer Hamiltonian layer

$$
U_M(\beta) = e^{-i \beta H_M}
\qquad\text{where } H_M = \sum_i X_i
$$

### Full QAOA circuit (depth \(p\)):

$$
|\psi(\gamma, \beta)\rangle
=
U_M(\beta_p)
U_C(\gamma_p)
\cdots
U_M(\beta_1)
U_C(\gamma_1)
|+\rangle^{\otimes n}
$$

- qubits represent binary decision variables \(x_{i,m}\)  
- initial state: \(|+\rangle^{\otimes n}\)  
- cost unitaries implement \(e^{-i\gamma J_{i,j} Z_i Z_j}\) and \(e^{-i\gamma h_i Z_i}\)  
- mixing unitaries are \(R_X(2\beta)\)  
- measure in Z basis to obtain assignments

### Initial Superposition

$$
|\psi_0\rangle
=
|+\rangle^{\otimes n}
=
\frac{1}{\sqrt{2^n}}
\sum_{x \in \{0,1\}^n} |x\rangle
$$

### QAOA State

$$
|\psi(\gamma, \beta)\rangle
=
U_M(\beta) U_C(\gamma) |\psi_0\rangle
$$

### Probability of measuring bitstring \(x\):

$$
P(x)
=
\left| \langle x \mid \psi(\gamma, \beta) \rangle \right|^2
$$

### Optimal Parameters

$$
(\gamma^*, \beta^*) =
\arg\min_{\gamma,\beta}
\langle
\psi(\gamma, \beta)
\mid
H_C
\mid
\psi(\gamma, \beta)
\rangle
$$
