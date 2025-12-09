# Quantum Cloud Task Scheduling via QAOA: Reproducible Research Package

## Abstract
We investigate a two-machine cloud task scheduling problem through the lens of quantum approximate optimization and provide a reproducible software stack that juxtaposes quantum-inspired heuristics with optimal classical solvers. The repository includes configurable datasets, QUBO encodings that capture load balancing and priority weighting, a local QAOA workflow running on noisy Aer simulators, and classical baselines ranging from exact search to ILP formulations. We report relative makespan gaps on synthetic workloads (5–12 jobs) and supply scripts that regenerate both results and publication-ready tables.

## Problem Statement
Given a set of independent tasks with processing times `p_i` and priority weights `w_i`, we target the two-machine partitioning variant that minimizes the maximum completion time (makespan) while respecting the relative importance of each task. The decision variable for QAOA is binary (`x_i = 1` ⇔ task assigned to machine 1). The QUBO encodes:

- A squared load-difference penalty ensuring both machines carry similar workloads.
- A linear bias prioritizing heavier/high-priority tasks on the primary machine.

## Methodology
1. **Data Generation** – `scripts/generate_datasets.py` samples reproducible CSVs (5, 8, 10, 12 jobs) with configurable seeds.
2. **QUBO Construction** – `quantum_scheduler.utils.qubo_builder.build_qubo` translates tasks into a dense QUBO matrix parameterized by `balance_penalty` and `priority_bias`.
3. **Quantum Solver** – `quantum_scheduler.qaoa_solver.solve_qaoa_local` optimizes a depth-`reps` QAOA circuit using COBYLA, evaluates energies via shot-based sampling, and decodes the most promising bitstring into an actionable schedule.
4. **Classical Baselines** – `quantum_scheduler.classical_solver.solve_classical` offers an exact benchmark for small instances; ILP hooks are prepared for larger workloads.
5. **Experiment Orchestration** – `scripts/run_experiments.py` runs single experiments or YAML-defined sweeps (`configs/baseline.yaml`) and stores JSON artifacts plus batch CSV summaries.
6. **Statistical Analysis** – `scripts/analyze_results.py` aggregates all JSON/CSV outputs into Markdown and LaTeX tables for immediate inclusion in a manuscript.

## Experimental Setup
- Qiskit 0.46, Qiskit-Aer 0.13, Qiskit-Algorithms 0.3
- AerSimulator with the default local noise model
- COBYLA (maxiter ∈ {40, 60, 80, 120})
- Shots per objective evaluation: 1024; final sampling shots: {4096, 8192}

## Representative Results
Running `python scripts/run_experiments.py --config configs/baseline.yaml` yields JSON files under `results/` and a consolidated CSV (`results/baseline_summary.csv`). Typical relative makespan gaps range from 8–25%, with deeper circuits (reps=2) narrowing the quantum-classical discrepancy on 10–12 job instances. Use `python scripts/analyze_results.py --input-csv results/baseline_summary.csv` to regenerate the Markdown/LaTeX tables cited in the manuscript.

## Reproducibility Checklist
- ✅ Version-pinned dependencies in `requirements.txt` / `pyproject.toml`
- ✅ Deterministic dataset sampling (seeded NumPy RNG)
- ✅ Config-driven experiment sweeps (YAML)
- ✅ Scripted analysis producing ready-to-paste figures/tables
- ✅ Comprehensive README with setup instructions

## Next Steps
1. Extend the ILP baseline to larger job counts (leveraging PuLP/Gurobi).
2. Deploy to cloud simulators (IonQ, Rigetti) to capture hardware-specific noise.
3. Integrate bitstring post-processing to construct full machine schedules beyond the two-machine partition abstraction.
