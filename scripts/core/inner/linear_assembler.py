from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, TYPE_CHECKING, Dict

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
    return (q_blk - z_blk).astype(np.float32, copy=False).reshape(-1)


def _apply_dual_equiv(block_name: str, z: np.ndarray, z_prev: np.ndarray, eta: float, reg: "BlockRegistry") -> np.ndarray:
    z_blk = _view_block(reg, z, block_name)
    zp_blk = _view_block(reg, z_prev, block_name)
    return (eta * (z_blk - zp_blk)).astype(np.float32, copy=False).reshape(-1)


# ---------------------------------------------------------------------------
# Theory-aligned linear assembly (Table 5.1 row-blocks: S/C/Y/R)
# ---------------------------------------------------------------------------

_THEORY_GROUP_TO_BLOCK: Dict[str, str] = {
    "S": "f_hat",   # edge-flow consistency
    "C": "B_hat",   # budget aggregation
    "Y": "y_hat",   # average coverage
    "R": "r_hat",   # macro statistics
}


def _theory_groups_present(reg: "BlockRegistry") -> List[str]:
    """Return theory row-block names applicable for the current registry."""
    names = set(reg.names())
    out: List[str] = []
    for g in ("S", "C", "Y", "R"):
        blk = _THEORY_GROUP_TO_BLOCK[g]
        if blk in names:
            out.append(g)
    return out


def _apply_primal_theory(group: str, q: np.ndarray, z: np.ndarray, reg: "BlockRegistry") -> np.ndarray:
    """Consensus-equivalent proxy for theory row-blocks."""
    blk = _THEORY_GROUP_TO_BLOCK.get(group, "")
    if not blk or blk not in set(reg.names()):
        return np.zeros((0,), dtype=np.float32)
    return _apply_primal_equiv(blk, q, z, reg)


def _apply_dual_theory(group: str, z: np.ndarray, z_prev: np.ndarray, eta: float, reg: "BlockRegistry") -> np.ndarray:
    """Dual proxy for theory row-blocks: eta*(z_blk - z_prev_blk)."""
    blk = _THEORY_GROUP_TO_BLOCK.get(group, "")
    if not blk or blk not in set(reg.names()):
        return np.zeros((0,), dtype=np.float32)
    return _apply_dual_equiv(blk, z, z_prev, float(eta), reg)


def build_linear_assembly(problem: "CadmmProblem", reg: "BlockRegistry", flags: "FeatureFlags") -> LinearAssembly:
    """Build LinearAssembly for LinearResidual.

    Stage C default is consensus-equivalent.

    Theory mode switches equality-consensus reporting to row-blocks S/C/Y/R (Table 5.1).
    """
    use_coupled = bool(getattr(flags, "use_coupled_linear_assembly", False))
    mode = str(getattr(flags, "linear_assembly_mode", "consensus_equiv"))
    enable_coupled_groups = bool(getattr(flags, "enable_coupled_groups", False))
    log_groups = bool(getattr(flags, "log_linear_groups", False)) or bool(
        getattr(flags, "log_linear_residual_groups", False)
    )

    params = getattr(problem, "params", None)
    coupled_scale = float(getattr(params, "coupled_residual_scale", 1.0))
    diag_scale = float(getattr(params, "diag_group_scale", 1.0))

    theory_mode = bool(getattr(flags, "enable_theory_mode", False))
    enable_y_avg_assembly = bool(getattr(flags, "enable_y_avg_assembly", False))

    # Default: macro block residual groups.
    # Theory: replace equality macro blocks with S/C/Y/R to avoid double counting.
    base_names = list(reg.names())
    theory_names: List[str] = []
    if theory_mode:
        eq_blks = {"f_hat", "B_hat", "y_hat", "r_hat"}
        base_names = [bn for bn in base_names if bn not in eq_blks]
        theory_names = _theory_groups_present(reg)

    coupled_names: list[str] = []
    if enable_coupled_groups and mode in ("coupled_basic", "assembled_ops"):
        from scripts.core.inner import coupled_groups
        coupled_names = list(coupled_groups.list_coupled_groups(problem, reg, flags))

    diag_names: list[str] = []
    if log_groups:
        diag_names = ["diag_sigma_polytope_violation", "diag_budget_violation"]

    names = base_names + theory_names + coupled_names + diag_names

    def _diag_violation_vec(group: str, z_vec: np.ndarray, reg2: "BlockRegistry") -> np.ndarray:
        if group == "diag_budget_violation":
            v = float(getattr(getattr(problem, "window", None), "budget_violation", 0.0))
            return np.asarray([v], dtype=np.float32)

        from scripts.core.inner.constraints import build_constraints
        from scripts.core.inner.projections import (
            viol_box, viol_simplex, viol_weighted_simplex, viol_polytope, viol_halfspace, viol_affine
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
            if isinstance(c, BoxC):
                viols.append(float(viol_box(x, c.lo, c.hi))); continue
            if isinstance(c, SimplexC):
                viols.append(float(viol_simplex(x, float(c.s)))); continue
            if isinstance(c, WeightedSimplexC):
                viols.append(float(viol_weighted_simplex(x, c.w, float(c.s)))); continue
            if isinstance(c, HalfspaceC):
                viols.append(float(viol_halfspace(x, c.a, float(c.b)))); continue
            if isinstance(c, AffineC):
                viols.append(float(viol_affine(x, c.A, c.b))); continue
            if isinstance(c, PolytopeC):
                viols.append(float(viol_polytope(x, c.G, c.h, c.lo, c.hi))); continue
            viols.append(0.0)

        v = np.asarray(viols, dtype=np.float32)
        return v if v.size > 0 else np.zeros((1,), dtype=np.float32)

    def apply_primal(name: str, q: np.ndarray, z: np.ndarray, reg2: "BlockRegistry") -> np.ndarray:
        # ------------------------------------------------------------------
        # YAVG override for theory row-block "Y":
        #   rY = mean(s_hat) - y_hat
        # This makes diagnostic curves consistent with the solver's stop criterion
        # under average-assembly. (Hard constraint B: do NOT use q_y - z_y.)
        # ------------------------------------------------------------------
        if enable_y_avg_assembly and name == "Y" and ("s_hat" in set(reg2.names())) and ("y_hat" in set(reg2.names())):
            N = int(getattr(problem, "N", 0) or problem.N)
            y = _view_block(reg2, z, "y_hat").astype(np.float32, copy=False).reshape(-1)  # (dy,)
            # s_hat block shape is (N, dy). In q (N,D), view becomes (N, N, dy).
            s_view = _view_block(reg2, q, "s_hat").astype(np.float32, copy=False)         # (N, N, dy)
            dy = int(y.size)
            # owned rows: robot i owns row i of its stacked block
            s_rows = np.zeros((N, dy), dtype=np.float32)
            for i in range(N):
                s_rows[i] = np.asarray(s_view[i, i, :], dtype=np.float32).reshape(-1)
            mean_s = np.mean(s_rows, axis=0).astype(np.float32, copy=False)
            return (mean_s - y).astype(np.float32, copy=False).reshape(-1)

        if name in _THEORY_GROUP_TO_BLOCK:
            # Optional: reuse Stage-D owner-masked assembly to partially realize
            # "undirected-edge orientation de-dup" semantics (Table 5.1).
            blk = _THEORY_GROUP_TO_BLOCK[name]
            if use_coupled and mode == "assembled_ops":
                from scripts.core.inner.assembled_ops import owner_mask_for_block
                q_blk = _view_block(reg2, q, blk)
                z_blk = _view_block(reg2, z, blk)
                q_flat = np.asarray(q_blk, dtype=np.float32).reshape((problem.N, -1))
                z_flat = np.asarray(z_blk, dtype=np.float32).reshape((-1,))
                own = owner_mask_for_block(blk, problem, reg2, flags)
                r = (q_flat - z_flat[None, :]) * own.astype(np.float32)
                return (coupled_scale * r.reshape(-1)).astype(np.float32)
            return _apply_primal_theory(name, q, z, reg2)

        if name.startswith("coupled_"):
            from scripts.core.inner import coupled_groups
            return coupled_groups.apply_coupled_primal(name, q, z, problem, reg2, flags)

        if name.startswith("diag_"):
            return (diag_scale * _diag_violation_vec(name, z, reg2)).astype(np.float32)

        if use_coupled and mode == "assembled_ops":
            from scripts.core.inner.assembled_ops import owner_mask_for_block
            q_blk = _view_block(reg2, q, name)
            z_blk = _view_block(reg2, z, name)
            q_flat = np.asarray(q_blk, dtype=np.float32).reshape((problem.N, -1))
            z_flat = np.asarray(z_blk, dtype=np.float32).reshape((-1,))
            own = owner_mask_for_block(name, problem, reg2, flags)
            r = (q_flat - z_flat[None, :]) * own.astype(np.float32)
            return (coupled_scale * r.reshape(-1)).astype(np.float32)

        return _apply_primal_equiv(name, q, z, reg2)

    def apply_dual(name: str, z: np.ndarray, z_prev: np.ndarray, eta: float, reg2: "BlockRegistry") -> np.ndarray:
        # ------------------------------------------------------------------
        # YAVG override for theory row-block "Y" dual diagnostic:
        #   sY = eta_y * (y_hat - y_hat_prev) / sqrt(N)
        # Notes:
        #   - This is a DIAGNOSTIC proxy consistent with the solver's stop definition.
        #   - Do NOT fall back to eta*(z_z - z_prev) or q_y - z_y.
        # ------------------------------------------------------------------
        if enable_y_avg_assembly and name == "Y" and ("y_hat" in set(reg2.names())):
            N = int(getattr(problem, "N", 0) or problem.N)
            y = _view_block(reg2, z, "y_hat").astype(np.float32, copy=False).reshape(-1)
            yp = _view_block(reg2, z_prev, "y_hat").astype(np.float32, copy=False).reshape(-1)
            # Prefer a fixed eta_y if solver stored it (keeps exact consistency under eta-balancing).
            params = getattr(problem, "params", None)
            eta_base = float(getattr(params, "eta", float(eta))) if params is not None else float(eta)
            eta_y_scale = float(getattr(params, "eta_y_avg_scale", 1.0)) if params is not None else 1.0
            eta_y = float(getattr(problem, "_eta_y_fixed", eta_y_scale * eta_base))
            if (not np.isfinite(eta_y)) or eta_y <= 0.0:
                eta_y = eta_y_scale * eta_base
            diff = (y - yp).astype(np.float32, copy=False)
            return (eta_y * diff / np.sqrt(float(N))).astype(np.float32, copy=False).reshape(-1)
        
        if name in _THEORY_GROUP_TO_BLOCK:
            return _apply_dual_theory(name, z, z_prev, eta, reg2)

        if name.startswith("coupled_"):
            from scripts.core.inner import coupled_groups
            return coupled_groups.apply_coupled_dual(name, z, z_prev, eta, problem, reg2, flags)

        if name.startswith("diag_"):
            v = _diag_violation_vec(name, z, reg2)
            return np.zeros_like(v, dtype=np.float32)

        return _apply_dual_equiv(name, z, z_prev, eta, reg2)

    return LinearAssembly(names=names, apply_primal=apply_primal, apply_dual=apply_dual)