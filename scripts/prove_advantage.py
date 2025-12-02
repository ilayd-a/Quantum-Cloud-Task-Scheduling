"""Demonstrate practical advantages of QAOA over classical heuristics."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.append(str(ROOT / "src"))

from quantum_scheduler import solve_qaoa_local  # noqa: E402
from quantum_scheduler.classical_solver import (  # noqa: E402
    solve_classical,
    solve_greedy,
)


def make_tasks(processing_times, priority=None):
    if priority is None:
        priority = [1] * len(processing_times)
    return [
        {"job": idx + 1, "p_i": float(p), "priority_w": float(w)}
        for idx, (p, w) in enumerate(zip(processing_times, priority))
    ]


def run_qaoa(tasks, **kwargs):
    return solve_qaoa_local(
        tasks,
        reps=kwargs.get("reps", 3),
        maxiter=kwargs.get("maxiter", 100),
        shots=kwargs.get("shots", 0),
        final_shots=kwargs.get("final_shots", 2048),
        balance_penalty=kwargs.get("balance_penalty"),
        priority_bias=kwargs.get("priority_bias", 0.0),
        optimizer=kwargs.get("optimizer", "spsa"),
        restarts=kwargs.get("restarts", 10),
        backend_type=kwargs.get("backend", "statevector"),
        seed=kwargs.get("seed"),
    )


def test_quality():
    print("=== TEST 1: Quality (Trap Dataset) ===")
    trap_tasks = make_tasks([10, 10, 20])

    greedy_res = solve_greedy([10, 10, 20], M=2)
    print(f"Greedy loads: {greedy_res['loads']}, makespan={greedy_res['makespan']}")

    qaoa_res = run_qaoa(
        trap_tasks,
        reps=2,
        maxiter=150,
        restarts=20,
        seed=7,
    )
    best_sched = qaoa_res["best_schedule"]
    print(
        f"QAOA loads: {best_sched['loads']}, makespan={best_sched['makespan']}, evaluations={qaoa_res['evaluations']}"
    )


def test_efficiency_and_diversity():
    print("\n=== TEST 2 & 3: Efficiency + Diversity (Random 10-job Dataset) ===")
    rng = random.Random(42)
    processing = [rng.randint(5, 20) for _ in range(10)]
    priority = [rng.randint(1, 3) for _ in range(10)]
    tasks = make_tasks(processing, priority)

    greedy = solve_greedy(processing, M=2)
    print(f"Greedy makespan: {greedy['makespan']}")

    qaoa_res = run_qaoa(
        tasks,
        reps=3,
        maxiter=200,
        restarts=25,
        seed=21,
    )
    qaoa_sched = qaoa_res["best_schedule"]
    print(
        f"QAOA makespan: {qaoa_sched['makespan']} (loads={qaoa_sched['loads']}), evaluations={qaoa_res['evaluations']}"
    )

    brute_force_space = 2 ** len(processing)
    efficiency = brute_force_space / max(1, qaoa_res["evaluations"])
    print(f"Search efficiency ratio: {efficiency:.2f} x fewer evaluations than brute force")

    print("\nTop 3 schedules discovered by QAOA:")
    for idx, sol in enumerate(qaoa_res.get("top_solutions", []), start=1):
        print(
            f"  #{idx}: bitstring={sol['bitstring']}, energy={sol['energy']:.2f}, "
            f"makespan={sol['makespan']}, loads={sol['loads']}"
        )


def main():
    test_quality()
    test_efficiency_and_diversity()


if __name__ == "__main__":
    main()
