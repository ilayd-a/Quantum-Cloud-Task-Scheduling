from itertools import product
from typing import Iterable, Sequence

try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False


def solve_classical(p: Sequence[int], M: int = 3):
    """Brute-force classical baseline for the M-machine scheduling problem."""
    n = len(p)
    best_makespan = float("inf")
    best_assignment = None
    best_loads: Iterable[int] | None = None

    for assignment in product(range(M), repeat=n):
        loads = [0] * M
        for i, m in enumerate(assignment):
            loads[m] += p[i]

        makespan = max(loads)

        if makespan < best_makespan:
            best_makespan = makespan
            best_assignment = assignment
            best_loads = loads.copy()

    if best_loads is None:
        best_loads = [0] * M

    return {
        "assignment": best_assignment,
        "loads": [int(l) for l in best_loads],
        "makespan": int(best_makespan),
        "method": "brute_force",
    }


def solve_greedy(p: Sequence[int], M: int = 2):
    """
    Simple greedy heuristic: assign each task to the machine with the lowest current load.
    """
    loads = [0] * M
    assignment = []

    for duration in p:
        machine = min(range(M), key=lambda m: loads[m])
        loads[machine] += duration
        assignment.append(machine)

    return {
        "assignment": assignment,
        "loads": loads,
        "makespan": max(loads),
        "method": "greedy",
    }


def solve_lpt(p: Sequence[int], M: int = 2):
    """
    Longest Processing Time first (LPT) heuristic.
    
    Sort jobs in descending order by processing time, then assign each to the
    machine with the lowest current load. This is a standard scheduling heuristic
    with good approximation guarantees.
    """
    # Sort jobs by processing time (descending) with original indices
    indexed_p = list(enumerate(p))
    indexed_p.sort(key=lambda x: x[1], reverse=True)
    
    loads = [0] * M
    assignment = [0] * len(p)
    
    for original_idx, duration in indexed_p:
        # Assign to machine with lowest load
        machine = min(range(M), key=lambda m: loads[m])
        loads[machine] += duration
        assignment[original_idx] = machine
    
    return {
        "assignment": assignment,
        "loads": loads,
        "makespan": max(loads),
        "method": "lpt",
    }


def solve_ilp(p: Sequence[int], M: int = 2, solver: str | None = None):
    """
    Integer Linear Programming formulation for the makespan minimization problem.
    
    Formulation:
    - Decision variables: x[i][m] = 1 if task i assigned to machine m, else 0
    - Objective: minimize makespan (maximum load)
    - Constraints: each task assigned to exactly one machine
    
    Args:
        p: Processing times for each task
        M: Number of machines
        solver: PuLP solver name (None = default, 'CBC' = COIN-OR, 'GUROBI' = Gurobi, etc.)
    
    Returns:
        Dictionary with assignment, loads, makespan, and method='ilp'
    """
    if not PULP_AVAILABLE:
        raise ImportError(
            "PuLP is required for ILP solver. Install with: pip install pulp"
        )
    
    n = len(p)
    if n == 0:
        return {
            "assignment": [],
            "loads": [0] * M,
            "makespan": 0,
            "method": "ilp",
        }
    
    # Create problem
    prob = pulp.LpProblem("MakespanMinimization", pulp.LpMinimize)
    
    # Decision variables: x[i][m] = 1 if task i goes to machine m
    x = {}
    for i in range(n):
        for m in range(M):
            x[i, m] = pulp.LpVariable(f"x_{i}_{m}", cat='Binary')
    
    # Makespan variable (upper bound on all machine loads)
    makespan = pulp.LpVariable("makespan", lowBound=0, cat='Continuous')
    
    # Objective: minimize makespan
    prob += makespan
    
    # Constraints: each task assigned to exactly one machine
    for i in range(n):
        prob += sum(x[i, m] for m in range(M)) == 1, f"Task_{i}_assignment"
    
    # Constraints: makespan >= load on each machine
    for m in range(M):
        prob += makespan >= sum(p[i] * x[i, m] for i in range(n)), f"Machine_{m}_load"
    
    # Solve
    if solver is None:
        solver_obj = pulp.PULP_CBC_CMD(msg=0)  # Silent mode
    else:
        solver_obj = solver
    
    try:
        prob.solve(solver_obj)
    except Exception:
        # Fallback to default solver
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    if prob.status != pulp.LpStatusOptimal:
        # Fallback to greedy if ILP fails
        return solve_greedy(p, M)
    
    # Extract solution
    assignment = []
    loads = [0.0] * M
    
    for i in range(n):
        for m in range(M):
            if pulp.value(x[i, m]) > 0.5:  # Binary variable
                assignment.append(m)
                loads[m] += p[i]
                break
        else:
            # Should not happen, but assign to machine 0 as fallback
            assignment.append(0)
            loads[0] += p[i]
    
    return {
        "assignment": assignment,
        "loads": [float(l) for l in loads],
        "makespan": float(pulp.value(makespan)),
        "method": "ilp",
        "status": pulp.LpStatus[prob.status],
    }
