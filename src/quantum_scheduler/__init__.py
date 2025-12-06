"""Top-level package for the quantum cloud task scheduling toolkit."""

__all__ = ["solve_classical", "solve_qaoa_local"]


def __getattr__(name):
    if name == "solve_classical":
        from .classical_solver import solve_classical as _solve_classical

        return _solve_classical
    if name == "solve_qaoa_local":
        from .qaoa_solver import solve_qaoa_local as _solve_qaoa_local

        return _solve_qaoa_local
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
