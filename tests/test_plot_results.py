import importlib.util
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "plot_results.py"
SPEC = importlib.util.spec_from_file_location("plot_results", MODULE_PATH)
plot_results = importlib.util.module_from_spec(SPEC)
sys.modules["plot_results"] = plot_results
assert SPEC.loader is not None
SPEC.loader.exec_module(plot_results)


def _sample_entry(tmp_path: Path) -> dict:
    dataset_path = tmp_path / "toy_dataset.csv"
    dataset_path.write_text("job,p_i,priority_w\n1,4,1\n2,6,1\n")

    return {
        "dataset": dataset_path.name,
        "dataset_path": str(dataset_path),
        "num_tasks": 2,
        "qaoa": {
            "energy": -1.0,
            "counts": {"00": 12, "01": 7, "10": 3},
            "energy_trace": [
                {"restart": 1, "evaluation": 1, "energy": 0.5},
                {"restart": 1, "evaluation": 2, "energy": 0.3},
            ],
            "balance_penalty_multiplier": 10.0,
            "priority_bias": 0.1,
        },
        "classical": {"makespan": 8},
        "metrics": {"qaoa_makespan": 9, "classical_makespan": 8},
        "config": {
            "reps": 1,
            "shots": 32,
            "final_shots": 128,
            "balance_penalty_multiplier": 10.0,
            "priority_bias": 0.1,
        },
    }


def test_qubo_plots(tmp_path):
    entry = _sample_entry(tmp_path)
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()

    heatmap = plot_results.plot_qubo_heatmap(entry, plots_dir)
    landscape = plot_results.plot_qubo_landscape(entry, plots_dir)

    assert heatmap is not None and heatmap.exists()
    assert landscape is not None and landscape.exists()


def test_sample_histogram_and_convergence(tmp_path):
    entry = _sample_entry(tmp_path)
    plots_dir = tmp_path / "plots_hist"
    plots_dir.mkdir()

    histogram = plot_results.plot_sample_histogram(entry, plots_dir)
    convergence = plot_results.plot_convergence_curves([entry], plots_dir)

    assert histogram is not None and histogram.exists()
    assert convergence is not None and convergence.exists()


def test_makespan_and_classical_plots(tmp_path):
    entry = _sample_entry(tmp_path)
    plots_dir = tmp_path / "plots_makespan"
    plots_dir.mkdir()

    makespan = plot_results.plot_makespan_comparison([entry], plots_dir)
    classical = plot_results.plot_classical([entry], plots_dir)
    qaoa_energy = plot_results.plot_qaoa_energy([entry], plots_dir)

    assert makespan is not None and makespan.exists()
    assert classical is not None and classical.exists()
    assert qaoa_energy is not None and qaoa_energy.exists()
