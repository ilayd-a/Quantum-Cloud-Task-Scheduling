from utils.dataset_loader import load_tasks
from qaoa_solver import solve_qaoa_local

dataset_path = "datasets/dataset_5.csv"
tasks = load_tasks(dataset_path)

qaoa_res = solve_qaoa_local(tasks, reps=1, maxiter=20)
print(qaoa_res)