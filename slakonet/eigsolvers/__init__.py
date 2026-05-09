from slakonet.eigsolvers.base import _EigSolver
from slakonet.eigsolvers.cholesky_eigh import CholeskyEighSolver
from slakonet.eigsolvers.lanczos import LanczosSolver
from slakonet.eigsolvers.vqe import VqeSolver

__all__ = ["_EigSolver", "CholeskyEighSolver", "LanczosSolver", "VqeSolver", "make_eigsolver"]


def make_eigsolver(cfg) -> _EigSolver:
    """Instantiate an eigensolver from a config dataclass."""
    name = cfg.solver_name
    if name == "cholesky_eigh":
        return CholeskyEighSolver(cfg)
    elif name == "lanczos":
        return LanczosSolver(cfg)
    elif name == "vqe":
        return VqeSolver(cfg)
    else:
        raise ValueError(f"Unknown eigsolver solver_name: {name!r}")
