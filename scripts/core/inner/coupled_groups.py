from __future__ import annotations

"""Stage E: cross-block coupled residual groups.

This module defines residual *groups* that depend on multiple blocks (or window
state) and can be optionally included in the stopping / residual balancing
total norms.

All options are accessed via getattr(flags/params). No dataclass fields are
required to exist.
"""

from typing import List

import numpy as np


def list_coupled_groups(problem: object, reg: object, flags: object) -> List[str]:
    """Return enabled coupled group names.

    Enabled iff:
      - getattr(flags, "enable_coupled_groups", False) is True
      - getattr(flags, "linear_assembly_mode", "consensus_equiv") in
        {"coupled_basic", "assembled_ops"}
    """
    _ = (problem, reg)
    if not bool(getattr(flags, "enable_coupled_groups", False)):
        return []
    mode = str(getattr(flags, "linear_assembly_mode", "consensus_equiv"))
    if mode not in ("coupled_basic", "assembled_ops"):
        return []
    return [
        "coupled_sigma_y",
        "coupled_flow_vs_budget_edge",
    ]


def group_dim(name: str, problem: object, reg: object, flags: object) -> int:
    """Return the residual vector dimension for a given coupled group."""
    _ = (reg, flags)
    if name == "coupled_sigma_y":
        return int(getattr(problem, "G", 0))
    if name == "coupled_flow_vs_budget_edge":
        return int(getattr(problem, "E", 0))
    return 0


def _source_vector(q: np.ndarray, z: np.ndarray, flags: object) -> np.ndarray:
    src = str(getattr(flags, "coupled_primal_source", "z"))
    if src == "q_mean":
        q_arr = np.asarray(q, dtype=np.float32)
        if q_arr.ndim == 2 and q_arr.shape[0] > 0:
            return np.asarray(np.mean(q_arr, axis=0), dtype=np.float32).reshape(-1)
    return np.asarray(z, dtype=np.float32).reshape(-1)


def apply_coupled_primal(
    name: str,
    q: np.ndarray,
    z: np.ndarray,
    problem: object,
    reg: object,
    flags: object,
) -> np.ndarray:
    """Compute primal residual vector for a coupled group."""
    params = getattr(problem, "params", None)
    z_src = _source_vector(q, z, flags)
    reg_names = set(getattr(reg, "names")())

    if name == "coupled_sigma_y":
        G = int(getattr(problem, "G", 0))
        if G <= 0:
            return np.zeros((0,), dtype=np.float32)
        if ("sigma" not in reg_names) or ("y_hat" not in reg_names):
            return np.zeros((G,), dtype=np.float32)
        blocks = getattr(reg, "unpack")(z_src)
        sigma = np.asarray(blocks.get("sigma", np.zeros((G,), dtype=np.float32)), dtype=np.float32).reshape(G)
        y_hat = np.asarray(blocks.get("y_hat", np.zeros((G,), dtype=np.float32)), dtype=np.float32).reshape(G)

        from scripts.core.inner.coverage_metrics import sigma_lower_bound_from_y_hat

        lo = sigma_lower_bound_from_y_hat(y_hat, params=params, flags=flags)
        lo = np.asarray(lo, dtype=np.float32).reshape(G)
        r = np.maximum(lo - sigma, 0.0)
        scale = float(getattr(params, "coupled_sigma_y_scale", 1.0))
        if not np.isfinite(scale):
            scale = 1.0
        return (scale * r).astype(np.float32, copy=False)

    if name == "coupled_flow_vs_budget_edge":
        E = int(getattr(problem, "E", 0))
        if E <= 0:
            return np.zeros((0,), dtype=np.float32)
        if ("f_hat" not in reg_names) or ("B_hat" not in reg_names):
            return np.zeros((E,), dtype=np.float32)
        blocks = getattr(reg, "unpack")(z_src)
        f_hat = np.asarray(blocks.get("f_hat", np.zeros((E,), dtype=np.float32)), dtype=np.float32).reshape(E)
        B_hat = np.asarray(blocks.get("B_hat", np.zeros((E,), dtype=np.float32)), dtype=np.float32).reshape(E)
        r = np.maximum(f_hat - B_hat, 0.0)
        scale = float(getattr(params, "coupled_flow_budget_scale", 1.0))
        if not np.isfinite(scale):
            scale = 1.0
        return (scale * r).astype(np.float32, copy=False)

    dim = group_dim(name, problem, reg, flags)
    return np.zeros((dim,), dtype=np.float32)


def apply_coupled_dual(
    name: str,
    z: np.ndarray,
    z_prev: np.ndarray,
    eta: float,
    problem: object,
    reg: object,
    flags: object,
) -> np.ndarray:
    """Compute dual residual vector for a coupled group.

    Default to a stable zero vector (dual coupling not enforced by default).
    """
    _ = (z, z_prev, eta)
    dim = group_dim(name, problem, reg, flags)
    if dim <= 0:
        return np.zeros((0,), dtype=np.float32)
    return np.zeros((dim,), dtype=np.float32)
