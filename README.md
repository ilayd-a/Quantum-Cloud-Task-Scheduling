# Quantum Cloud Task Scheduling

A small research sandbox for comparing classical scheduling baselines against a QAOA-style quantum formulation of the same cloud task allocation problem. The repo now follows a simple `src/` layout so it can be installed as a Python package or executed via the helper scripts under `scripts/`.

## Project Layout
```
├── data/
│   └── datasets/            # CSV inputs (re-generated via scripts)
├── results/
│   └── plots/               # Figures created by plot_results.py
├── scripts/
│   ├── generate_datasets.py # Utility to sample synthetic instances
│   ├── run_experiments.py   # Runs QAOA + classical baseline
│   └── plot_results.py      # Consumes JSON outputs and makes plots
├── src/
│   └── quantum_scheduler/
│       ├── classical_solver.py
│       ├── qaoa_solver.py
│       └── utils/
└── pyproject.toml / requirements.txt
```

## Getting Started
1. **Create a virtual environment** (Python ≥ 3.10) and install the package in editable mode:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -e .
   ```
2. **Regenerate datasets (optional).** The repo ships with a few CSVs, but you can quickly refresh them:
   ```bash
   python scripts/generate_datasets.py
   ```
3. **Run experiments.** This executes the QAOA pipeline plus the classical brute-force baseline and stores a JSON artifact in `results/`:
   ```bash
   python scripts/run_experiments.py --dataset dataset_5.csv --reps 1 --maxiter 20
   ```
   Use `--dataset /abs/path/to/file.csv` to point at custom inputs, and `--machines` to change the classical baseline configuration.
4. **Plot outcomes.** After generating multiple result files (e.g., one per dataset size), produce summary plots:
   ```bash
   python scripts/plot_results.py
   ```

## Core Modules
- `quantum_scheduler.classical_solver.solve_classical`: brute-force M-machine scheduling baseline. Returns best assignment, machine loads, and makespan.
- `quantum_scheduler.qaoa_solver.solve_qaoa_local`: builds a diagonal QUBO from task weights, converts it to an Ising Hamiltonian, and optimizes a QAOA circuit on local Aer noise models.
- `quantum_scheduler.utils`: dataset generation/loading helpers, state decoder, and the corrected QUBO builder that accepts either `(p, w)` or `(p_i, priority_w)` task dictionaries.

## Results Artifacts
Each invocation of `run_experiments.py` writes a JSON blob shaped like:
```json
{
  "dataset": "dataset_5.csv",
  "num_tasks": 5,
  "qaoa": {"energy": -9.25, "optimal_params": [0.41, 0.87]},
  "classical": {"assignment": [0, 1, 0, 2, 1], "loads": [14, 12, 9], "makespan": 14}
}
```
`plot_results.py` scans every `*_results.json` file in `results/` and emits PNGs under `results/plots/` for both the classical makespan trend and the QAOA energy trend.

## Notes & Limitations
- `qiskit>=0.46` (or Terra 2.x) is required because the solver relies on `BackendSamplerV2`. Ensure that `qiskit-aer` and `qiskit-algorithms` are also installed.
- The current QAOA workflow reports the Hamiltonian expectation value (“energy”). Translating that into an actual schedule or makespan would require additional bitstring post-processing, which is left as future work.
- Generated datasets and result artifacts are ignored by git so you can freely iterate without polluting the repo.
