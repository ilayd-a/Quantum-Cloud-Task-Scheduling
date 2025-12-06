# Quantum-Cloud-Task-Scheduling

Implementation of a toy scheduling problem (2 machines, 4+ tasks) with classical baselines and QUBO/Ising encoding as the foundation for exploring quantum optimization approaches.

## Quickstart

1. **Environment**  
   Ensure Python 3.10+ is available and install the minimal dependencies:

   ```bash
   pip install -r requirements.txt  # if available
   # or the direct set we currently use:
   pip install qiskit qiskit-aer qiskit-algorithms matplotlib pandas numpy
   ```

2. **Datasets**  
   Sample CSVs live in `datasets/` (e.g., `dataset_5.csv`). Regenerate or extend them with `python generate_all_datasets.py` if needed.

3. **Classical and QAOA runs**  
   Execute your experiment pipeline to populate `results/qaoa_results.json` and `results/classical_results.json`. A simple local QAOA run is provided via:

   ```bash
   python run_experiments.py
   ```

4. **Visualization suite**  
   After results exist, generate all plots (makespan bar chart, relative gap, QUBO landscape + heatmap, convergence curve, sample histogram) with:

   ```bash
   python plot_results.py
   ```

   Images are written to `results/plots/`.

5. **Tests (optional while they exist)**  
   Run `python -m pytest`. The current repo does not ship explicit tests yet, so Pytest will report “collected 0 items”.

## Gallery

<img width="468" height="500" alt="image" src="https://github.com/user-attachments/assets/341ee8c7-d680-4f3d-8e72-df6af84fb3b8" />

<img width="468" height="591" alt="image" src="https://github.com/user-attachments/assets/fbc07ee7-0b81-411f-98b2-ac3e5e52887c" />

<img width="468" height="475" alt="image" src="https://github.com/user-attachments/assets/5cb70377-1142-465a-89e0-78a1d17cfe9c" />

<img width="468" height="467" alt="image" src="https://github.com/user-attachments/assets/d49f0100-b8c0-4a8e-8b2e-2181f7e6bfaa" />

<img width="468" height="202" alt="image" src="https://github.com/user-attachments/assets/4be90b7d-4431-40b5-857b-eafdadcfb1c4" />

<img width="442" height="648" alt="image" src="https://github.com/user-attachments/assets/e06de47c-4da5-40ab-8a5d-44a76c2a1b7f" />
