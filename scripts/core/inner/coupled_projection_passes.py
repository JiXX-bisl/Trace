"""scripts.core.inner.coupled_projection_passes

Scheme-2 coupled feasibility in z-step.

Some z-step constraints are *cross-block*: a block's feasible region depends on
another block's value after projection. The solver already performs a first
pass of independent per-block projections. This module provides an optional
second pass that re-projects selected blocks using constraints rebuilt from the
already projected z-blocks.

Design goals:
- Keep the solver main loop intact (q / z / u / residual / stop / rb).
- All behaviors are gated by getattr(flags, ...) and default to Stage D.
- Robust to missing blocks (returns without raising).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from scripts.core.inner.blocks import BlockRegistry
from scripts.core.inner.constraints import build_constraints
from scripts.core.inner.projections import project_one_block, proj_y_sigma_polytope


def apply_coupled_projection_passes(
    problem: Any,
    reg: BlockRegistry,
    z_blocks: Dict[str, np.ndarray],
    diag_by_block: Dict[str, Dict[str, Any]],
    *,
    method: str,
    iters: int,
    tol: float,
) -> None:
    """Apply coupled z-step projections in-place.

    Parameters
    ----------
    problem:
        CadmmProblem-like snapshot (must have .flags).
    reg:
        BlockRegistry.
    z_blocks:
        Dict[str, ndarray] of **already projected** blocks (first-pass).
        This function may overwrite individual blocks.
    diag_by_block:
        Diagnostics dict from project_blocks; we will attach a "coupled_pass"
        entry to affected blocks.
    method/iters/tol:
        Projection configuration.
    """

    flags = getattr(problem, "flags", None)
    if flags is None:
        return

    # ------------------------------------------------------------------
    # Coupling 1: sigma depends on projected y_hat.
    # ------------------------------------------------------------------
    theory_mode = bool(getattr(flags, "enable_theory_mode", False))
    joint_poly = bool(getattr(flags, "enable_sigma_y_joint_polytope", False))
    if (theory_mode or joint_poly) and ("sigma" in z_blocks) and ("y_hat" in z_blocks):
        _reproject_y_sigma_joint(problem=problem, reg=reg, z_blocks=z_blocks, diag_by_block=diag_by_block, method=method, iters=iters, tol=tol)
    elif (
        bool(getattr(flags, "enable_sigma_coupled_to_y", True))
        and ("sigma" in z_blocks)
        and ("y_hat" in z_blocks)
    ):
        _reproject_one_block(problem, reg, z_blocks, diag_by_block, "sigma", method, iters, tol)

    # ------------------------------------------------------------------
    # Coupling 2 (optional): flow <= budget (edge-wise).
    # Enforced by tightening f_hat's upper bound using projected B_hat.
    # ------------------------------------------------------------------
    if (
        bool(getattr(flags, "enable_flow_coupled_to_budget", False))
        and ("f_hat" in z_blocks)
        and ("B_hat" in z_blocks)
    ):
        _reproject_one_block(problem, reg, z_blocks, diag_by_block, "f_hat", method, iters, tol)

def _merge_diag_coupled(
    diag_by_block: Dict[str, Dict[str, Any]],
    block_name: str,
    info: Dict[str, Any]
) -> None:
    """Attach coupled-pass diagnostics to a block, keeping max_violation monotone"""
    d0 = diag_by_block.get(
        block_name, 
        {"iters_used": 0, "violations": [], "max_violation": 0.0, "delta_norm": 0.0},
    )
    d0 = dict(d0)
    d0["coupled_pass"] = dict(info)
    d0["max_violation"] = float(
        max(
            float(d0.get("max_violation", 0.0) or 0.0),
            float(info.get("max_violation", 0.0) or 0.0),
        )
    )
    diag_by_block[block_name] = d0

def _extract_sigma_max_from_constraints(
    cons_sigma: Any,
    *,
    fallback: float = 1.0,
) -> Any:
    """Best-effort extract sigma_max from sigma's BoxC constraint (hi/ub/upper)."""
    if cons_sigma is None:
        return float(fallback)
    for c in cons_sigma:
        for hi_name in ("hi", "ub", "upper", "u"):
            if hasattr(c, hi_name):
                try:
                    return np.asarray(getattr(c, hi_name), dtype=np.float32).reshape(-1)
                except Exception:
                    continue
    return float(fallback)

def _reproject_y_sigma_joint(
    problem: Any,
    reg: BlockRegistry,
    z_blocks: Dict[str, np.ndarray],
    diag_by_block: Dict[str, Dict[str, Any]],
    method: str,
    iters: int,
    tol: float,
) -> None:
    """Jointly project (y_hat, sigma) onto the theory polytope P."""
    try:
        if ("y_hat" not in z_blocks) or ("sigma" not in z_blocks):
            return

        # Prefer sigma_max from assembled constraints (robust to scalar/vector sigma_max).
        cons2 = build_constraints(problem=problem, reg=reg, z_blocks=z_blocks)
        cons_sigma = cons2.get("sigma", [])
        smax = _extract_sigma_max_from_constraints(
            cons_sigma,
            fallback=float(getattr(getattr(problem, "params", None), "sigma_max", 1.0)),
        )

        y0 = np.asarray(z_blocks["y_hat"], dtype=np.float32)
        s0 = np.asarray(z_blocks["sigma"], dtype=np.float32)

        y1, s1, info = proj_y_sigma_polytope(
            y0, s0, smax, method=method, iters=iters, tol=tol, return_info=True
        )

        z_blocks["y_hat"] = np.asarray(y1, dtype=np.float32)
        z_blocks["sigma"] = np.asarray(s1, dtype=np.float32)

        # Normalize info keys to the same schema used by project_one_block diagnostics.
        info2 = dict(info)
        info2["op"] = "proj_y_sigma_polytope"
        info2["method"] = str(method)
        info2["max_violation"] = float(info.get("post_max_violation", 0.0) or 0.0)
        info2["pre_max_violation"] = float(info.get("pre_max_violation", 0.0) or 0.0)
        info2["delta_norm"] = float(info.get("corr_norm", 0.0) or 0.0)

        # Attach coupled-pass diag to both blocks (as requested).
        _merge_diag_coupled(diag_by_block, "y_hat", info2)
        _merge_diag_coupled(diag_by_block, "sigma", info2)
    except Exception:
        return


def _reproject_one_block(
    problem: Any,
    reg: BlockRegistry,
    z_blocks: Dict[str, np.ndarray],
    diag_by_block: Dict[str, Dict[str, Any]],
    block_name: str,
    method: str,
    iters: int,
    tol: float,
) -> None:
    """Re-project a single block using constraints built from current z_blocks."""

    if block_name not in z_blocks:
        return

    try:
        cons2 = build_constraints(problem=problem, reg=reg, z_blocks=z_blocks)
        cons = cons2.get(block_name, [])
        new_val, info = project_one_block(
            z_blocks[block_name],
            cons,
            method=method,
            iters=iters,
            tol=tol,
        )
        z_blocks[block_name] = np.asarray(new_val, dtype=np.float32)

        # Merge diagnostics (keep max_violation monotone, attach coupled_pass info).
        d0 = diag_by_block.get(
            block_name,
            {"iters_used": 0, "violations": [], "max_violation": 0.0, "delta_norm": 0.0},
        )
        d0 = dict(d0)
        d0["coupled_pass"] = info
        d0["max_violation"] = float(
            max(
                float(d0.get("max_violation", 0.0) or 0.0),
                float(info.get("max_violation", 0.0) or 0.0),
            )
        )
        diag_by_block[block_name] = d0
    except Exception:
        # Best-effort: coupled pass should never crash the solver.
        return
