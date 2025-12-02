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
