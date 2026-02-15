from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from scripts.core.data import CadmmProblem, FeatureFlags
    from scripts.core.inner.blocks import BlockRegistry


@dataclass
class LinearAssembly:
    """
    Engineering interface for Linear Assembled residuals

    Equivalenrt to consensus residuals:
    - primal: vec(q_block - z_block)
    - dual:   vec(eta * (z_block - z_prev_block))

    TODO: inject coupling across blocks at the apply_* hooks.
    """
    names: List[str]
    apply_primal: Callable[[str, np.ndarray, np.ndarray, "BlockRegistry"], np.ndarray]
    apply_dual:   Callable[[str, np.ndarray, np.ndarray, float, "BlockRegistry"], np.ndarray]

def _view_block(reg: "BlockRegistry", x: np.ndarray, name: str) -> np.ndarray:
    """
    Return a reshaped view/array of the given block.

    Supports:
      - x shape (D,)   -> returns shape reg.shape(name)
      - x shape (N,D)  -> returns shape (N,) + reg.shape(name)
    """
    sl = reg.sl(name)
    shp = reg.shape(name)
    x_arr = np.asarray(x)

    if x_arr.ndim == 1:
        if x_arr.size != reg.total_dim:
            raise ValueError(f"x has size {x_arr.size}, expected {reg.total_dim}.")
        return x_arr[sl].reshape(shp)
    
    if x_arr.ndim == 2:
        if x_arr.shape[1] != reg.total_dim:
            raise ValueError(f"x has shape {x_arr.shape}, expected (N, {reg.total_dim}).")
        N = x_arr.shape[0]
        return x_arr[:, sl].reshape((N,) + shp)
    
    raise ValueError(f"x must be 1-D or 2-D, but got shape = {x_arr.shape}")

def _apply_primal_equiv(block_name: str, q: np.ndarray, z: np.ndarray, reg: "BlockRegistry") -> np.ndarray:
    # q: (N,D), z: (D,)
    q_blk = _view_block(reg, q, block_name)  # (N,) + shp
    z_blk = _view_block(reg, z, block_name)  # shp
    # broadcast z_blk over N
    return (q_blk - z_blk).astype(np.float32, copy=False).reshape(-1)

def _apply_dual_equiv(block_name: str, z: np.ndarray, z_prev: np.ndarray, eta: float, reg: "BlockRegistry") -> np.ndarray:
    z_blk = _view_block(reg, z, block_name)
    zp_blk = _view_block(reg, z_prev, block_name)
    return (eta * (z_blk - zp_blk)).astype(np.float32, copy=False).reshape(-1)


def build_linear_assembly(problem: "CadmmProblem", reg: "BlockRegistry", flags: "FeatureFlags") -> LinearAssembly:
    """Build LinearAssembly for LinearResidual.

    Stage C default is consensus-equivalent.

    Stage D can optionally enable owner-masked assembled residuals and
    attach extra diagnostic groups. All options are toggled via getattr.
    """

    use_coupled = bool(getattr(flags, "use_coupled_linear_assembly", False))
    mode = str(getattr(flags, "linear_assembly_mode", "consensus_equiv"))
    enable_coupled_groups = bool(getattr(flags, "enable_coupled_groups", False))
    # Default must keep Stage C behavior: no extra groups unless explicitly enabled.
    log_groups = bool(getattr(flags, "log_linear_groups", False)) or bool(
        getattr(flags, "log_linear_residual_groups", False)
    )

    params = getattr(problem, "params", None)
    coupled_scale = float(getattr(params, "coupled_residual_scale", 1.0))
    diag_scale = float(getattr(params, "diag_group_scale", 1.0))

    base_names = list(reg.names())

    coupled_names: list[str] = []
    if enable_coupled_groups and mode in ("coupled_basic", "assembled_ops"):
        from scripts.core.inner import coupled_groups

        coupled_names = list(coupled_groups.list_coupled_groups(problem, reg, flags))

    diag_names: list[str] = []
    if log_groups:
        diag_names = ["diag_sigma_polytope_violation", "diag_budget_violation"]

    # Order is fixed for test stability.
    names = base_names + coupled_names + diag_names

    def _diag_violation_vec(group: str, z_vec: np.ndarray, reg2: "BlockRegistry") -> np.ndarray:
        if group == "diag_budget_violation":
            # Budget violation is a runtime diagnostic produced by the QoS closed loop.
            v = float(getattr(getattr(problem, "window", None), "budget_violation", 0.0))
            return np.asarray([v], dtype=np.float32)

        # Lazy imports to avoid circular deps.
        from scripts.core.inner.constraints import build_constraints
        from scripts.core.inner.projections import (
            viol_box,
            viol_simplex,
            viol_weighted_simplex,
            viol_polytope,
            viol_halfspace,
            viol_affine,
        )
        from scripts.core.data import BoxC, SimplexC, WeightedSimplexC, HalfspaceC, AffineC, PolytopeC

        z_blocks = reg2.unpack(z_vec)
        try:
            cons_by_block = build_constraints(problem, reg2, z_blocks=z_blocks)
        except Exception:
            return np.zeros((1,), dtype=np.float32)

        blk = "sigma" if (group == "diag_sigma_polytope_violation" and "sigma" in z_blocks) else None
        if blk is None:
            return np.zeros((1,), dtype=np.float32)

        x = np.asarray(z_blocks[blk], dtype=np.float32)
        cons = cons_by_block.get(blk, [])
        if len(cons) == 0:
            return np.zeros((1,), dtype=np.float32)

        viols: list[float] = []
        for c in cons:
            # dataclass constraints live in scripts.core.data
            if isinstance(c, BoxC):
                viols.append(float(viol_box(x, c.lo, c.hi)))
                continue
            if isinstance(c, SimplexC):
                viols.append(float(viol_simplex(x, float(c.s))))
                continue
            if isinstance(c, WeightedSimplexC):
                viols.append(float(viol_weighted_simplex(x, c.w, float(c.s))))
                continue
            if isinstance(c, HalfspaceC):
                viols.append(float(viol_halfspace(x, c.a, float(c.b))))
                continue
            if isinstance(c, AffineC):
                viols.append(float(viol_affine(x, c.A, c.b)))
                continue
            if isinstance(c, PolytopeC):
                viols.append(float(viol_polytope(x, c.G, c.h, c.lo, c.hi)))
                continue
            viols.append(0.0)
        v = np.asarray(viols, dtype=np.float32)
        if v.size == 0:
            v = np.zeros((1,), dtype=np.float32)
        return v

    def apply_primal(name: str, q: np.ndarray, z: np.ndarray, reg2: "BlockRegistry") -> np.ndarray:
        if name.startswith("coupled_"):
            from scripts.core.inner import coupled_groups

            return coupled_groups.apply_coupled_primal(name, q, z, problem, reg2, flags)

        if name.startswith("diag_"):
            return (diag_scale * _diag_violation_vec(name, z, reg2)).astype(np.float32)

        if use_coupled and mode == "assembled_ops":
            from scripts.core.inner.assembled_ops import owner_mask_for_block

            q_blk = _view_block(reg2, q, name)  # (N,) + shp
            z_blk = _view_block(reg2, z, name)  # shp
            q_flat = np.asarray(q_blk, dtype=np.float32).reshape((problem.N, -1))
            z_flat = np.asarray(z_blk, dtype=np.float32).reshape((-1,))
            own = owner_mask_for_block(name, problem, reg2, flags)
            r = (q_flat - z_flat[None, :]) * own.astype(np.float32)
            return (coupled_scale * r.reshape(-1)).astype(np.float32)

        return _apply_primal_equiv(name, q, z, reg2)

    def apply_dual(name: str, z: np.ndarray, z_prev: np.ndarray, eta: float, reg2: "BlockRegistry") -> np.ndarray:
        if name.startswith("coupled_"):
            from scripts.core.inner import coupled_groups

            return coupled_groups.apply_coupled_dual(name, z, z_prev, eta, problem, reg2, flags)
        if name.startswith("diag_"):
            v = _diag_violation_vec(name, z, reg2)
            return np.zeros_like(v, dtype=np.float32)
        # keep dual residual stable and consensus-equivalent
        return _apply_dual_equiv(name, z, z_prev, eta, reg2)

    return LinearAssembly(names=names, apply_primal=apply_primal, apply_dual=apply_dual)
