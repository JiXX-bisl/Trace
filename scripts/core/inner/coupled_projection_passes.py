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
from scripts.core.inner.projections import project_one_block


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
    if (
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
