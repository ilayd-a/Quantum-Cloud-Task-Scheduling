import pandas as pd

def load_tasks(file_path):
    """
    Reads a CSV of the form:
        job,p_i,priority_w
        1,5,3
        ...
    Returns: list of dicts
    """
    df = pd.read_csv(file_path)

    required_cols = {"job", "p_i", "priority_w"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    tasks = df.to_dict(orient="records")
    return tasks
