"""Utility helpers for dataset generation/loading, decoding and QUBO building."""

__all__ = [
    "generate_dataset",
    "load_tasks",
    "decode_solution_vector",
    "bitstring_to_vector",
    "build_qubo",
    "qubo_from_tasks",
]


def __getattr__(name):
    if name == "generate_dataset":
        from .dataset_generator import generate_dataset as _generate_dataset

        return _generate_dataset
    if name == "load_tasks":
        from .dataset_loader import load_tasks as _load_tasks

        return _load_tasks
    if name == "decode_solution_vector":
        from .decoder import decode_solution_vector as _decode_solution_vector

        return _decode_solution_vector
    if name == "bitstring_to_vector":
        from .decoder import bitstring_to_vector as _bitstring_to_vector

        return _bitstring_to_vector
    if name == "build_qubo":
        from .qubo_builder import build_qubo as _build_qubo

        return _build_qubo
    if name == "qubo_from_tasks":
        from .qubo_builder import qubo_from_tasks as _qubo_from_tasks

        return _qubo_from_tasks
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
