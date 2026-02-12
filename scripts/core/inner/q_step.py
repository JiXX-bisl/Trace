"""scripts.core.inner.q_step

Stage A: q-step local sub-problem solver module.

This module externalizes the q-step (local variable update) from
``scripts.core.cadmm_solver`` into a **switchable** and **unit-testable** layer.

Stage A focuses on a robust, deterministic local decision for the ``pos`` macro
block via candidate enumeration, while keeping other blocks equal to ``z``.
This enables ablation and incremental engineering without breaking the strict
I/O protocol:

  * z: (D,)
  * q,u: (N,D)

The cost terms are gated by ``problem.flags`` (FeatureFlags). All flag/param
reads use ``getattr(..., default)`` to preserve backward compatibility with
older snapshots/tests that may not provide the new fields.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from scripts.core.data import CadmmProblem
from scripts.core.inner.blocks import BlockRegistry, decode_pos, set_block


# ---------------------------------------------------------------------------
# Cost terms (Stage A)
# ---------------------------------------------------------------------------


def cost_admm_quadratic(cand: np.ndarray, target: np.ndarray, eta: float) -> float:
    """0.5 * eta * ||cand - target||^2."""
    d = cand - target
    return 0.5 * float(eta) * float(np.dot(d, d))


def cost_move(cand: np.ndarray, current_pos: np.ndarray, move_w: float) -> float:
    """move_w * ||cand - current_pos||^2."""
    d = cand - current_pos
    return float(move_w) * float(np.dot(d, d))


def cost_explore_from_frontier_entropy(
    cand: np.ndarray,
    frontier_entropy: np.ndarray,
    new_w: float,
) -> float:
    """- new_w * entropy_at(cand) with safe bounds handling.

    Coordinate convention:
      - cand = (x, y)
      - frontier_entropy indexed as [row=y, col=x]

    Out-of-bounds -> entropy = 0.
    """
    ent = np.asarray(frontier_entropy)
    if ent.ndim != 2:
        return 0.0
    H, W = int(ent.shape[0]), int(ent.shape[1])
    x = int(np.round(float(cand[0])))
    y = int(np.round(float(cand[1])))
    if (x < 0) or (x >= W) or (y < 0) or (y >= H):
        e = 0.0
    else:
        e = float(ent[y, x])
    return -float(new_w) * e


def cost_task_distance(
    cand: np.ndarray,
    task_pos: np.ndarray,
    priority: np.ndarray,
    deadline: np.ndarray,
    urgency_w: float,
) -> float:
    """urgency_w * min_j (urgency_j * ||cand - task_pos_j||).

    urgency_j = priority_j / max(deadline_j, 1.0)
    If no tasks -> 0.
    """
    tp = np.asarray(task_pos, dtype=np.float32).reshape(-1, 2)
    if tp.shape[0] == 0:
        return 0.0
    pr = np.asarray(priority, dtype=np.float32).reshape(-1)
    dl = np.asarray(deadline, dtype=np.float32).reshape(-1)
    T = int(min(tp.shape[0], pr.size, dl.size))
    if T <= 0:
        return 0.0
    pr = pr[:T]
    dl = dl[:T]
    tp = tp[:T]

    urg = pr / np.maximum(dl, 1.0)
    d = tp - cand[None, :]
    dist = np.linalg.norm(d, axis=1)
    val = float(np.min(urg * dist))
    return float(urgency_w) * val


def cost_repulsion(
    cand: np.ndarray,
    current_pos: np.ndarray,
    repulsion_grad_i: np.ndarray,
    rep_w: float,
) -> float:
    """rep_w * dot(repulsion_grad[rid], cand-current_pos)."""
    grad = np.asarray(repulsion_grad_i, dtype=np.float32).reshape(2)
    delta = cand - current_pos
    return float(rep_w) * float(np.dot(grad, delta))


def pos_candidate_score(
    *,
    rid: int,
    cand: np.ndarray,
    target: np.ndarray,
    current_pos: np.ndarray,
    problem: CadmmProblem,
    eta: float,
) -> float:
    """Composable total cost for selecting a position candidate (Stage A)."""

    flags = getattr(problem, "flags", None)
    params = getattr(problem, "params", None)

    total = 0.0

    # ADMM quadratic term (default ON)
    if bool(getattr(flags, "enable_qstep_admm_term", True)):
        total += cost_admm_quadratic(cand, target, eta)

    # move
    if bool(getattr(flags, "enable_qstep_cost_move", False)):
        move_w = float(getattr(params, "move_w", 0.0))
        total += cost_move(cand, current_pos, move_w)

    # explore
    if bool(getattr(flags, "enable_qstep_cost_explore", False)):
        new_w = float(getattr(params, "new_w", 0.0))
        total += cost_explore_from_frontier_entropy(cand, problem.frontier_entropy, new_w)

    # task
    if bool(getattr(flags, "enable_qstep_cost_task", False)):
        urgency_w = float(getattr(params, "urgency_w", 0.0))
        task = getattr(problem, "task", None)
        if task is not None:
            total += cost_task_distance(
                cand,
                getattr(task, "task_pos", np.zeros((0, 2), dtype=np.float32)),
                getattr(task, "priority", np.zeros((0,), dtype=np.float32)),
                getattr(task, "deadline", np.zeros((0,), dtype=np.float32)),
                urgency_w,
            )

    # repulsion
    if bool(getattr(flags, "enable_qstep_cost_repulsion", False)):
        rep_w = float(getattr(params, "rep_w", 0.0))
        rep_grad = np.asarray(
            getattr(problem, "repulsion_grad", np.zeros((problem.N, 2), dtype=np.float32)),
            dtype=np.float32,
        )
        if rep_grad.ndim == 2 and rep_grad.shape[0] > rid:
            total += cost_repulsion(cand, current_pos, rep_grad[rid], rep_w)

    return float(total)


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------


def solve_local_q(
    rid: int,
    problem: CadmmProblem,
    reg: BlockRegistry,
    z: np.ndarray,
    u_i: np.ndarray,
    q_i_prev: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Solve local q-step for one robot (Stage A).

    Parameters
    ----------
    rid:
        Robot id in [0, N-1].
    problem:
        Frozen CadmmProblem snapshot.
    reg:
        BlockRegistry for the current problem.
    z:
        Consensus variable (D,).
    u_i:
        Scaled dual for this robot (D,).
    q_i_prev:
        Previous local q for this robot (D,). Stage A keeps this for API stability.
    rng:
        Random generator (unused in Stage A, kept for future stochastic tie-breaks).

    Returns
    -------
    np.ndarray
        q_i_new (D,). Baseline is ``z.copy()``, with ``pos`` updated if possible.
    """
    _ = q_i_prev  # reserved for Stage B extensions
    _ = rng

    # Baseline: copy z (keeps other blocks consistent with current pipeline behavior)
    q_i_new = np.asarray(z, dtype=np.float32).copy()

    # Robustness: if pos block absent -> no-op
    if "pos" not in reg.names():
        return q_i_new

    # Candidates
    try:
        candidates = np.asarray(problem.candidate_moves[rid], dtype=np.float32).reshape(-1, 2)
    except Exception:
        return q_i_new
    if candidates.shape[0] == 0:
        return q_i_new

    N = int(getattr(problem, "N", 0)) or int(problem.N)

    # Target from (z - u_i).pos
    try:
        target_all = decode_pos(
            reg,
            np.asarray(z, dtype=np.float32) - np.asarray(u_i, dtype=np.float32),
            N,
        )
    except Exception:
        # If something is wrong with u_i shape, fall back to z target
        target_all = decode_pos(reg, np.asarray(z, dtype=np.float32), N)
    target = np.asarray(target_all[rid], dtype=np.float32).reshape(2)

    current_pos = np.asarray(problem.robot_pos[rid], dtype=np.float32).reshape(2)

    eta = float(getattr(getattr(problem, "params", None), "eta", 1.0))

    # Evaluate candidates deterministically with (cost, idx) tie-break
    best_idx: int = 0
    best_cost: Optional[float] = None
    for idx in range(int(candidates.shape[0])):
        cand = candidates[idx]
        c = pos_candidate_score(
            rid=rid,
            cand=cand,
            target=target,
            current_pos=current_pos,
            problem=problem,
            eta=eta,
        )
        if best_cost is None:
            best_cost = c
            best_idx = idx
        else:
            # Strict tie-break on smaller index for determinism
            if (c < best_cost) or (c == best_cost and idx < best_idx):
                best_cost = c
                best_idx = idx

    best = np.asarray(candidates[best_idx], dtype=np.float32).reshape(2)

    # Update only pos block in q_i_new
    pos_all = np.asarray(decode_pos(reg, np.asarray(z, dtype=np.float32), N), dtype=np.float32)
    pos_all = pos_all.copy()  # decode_pos returns a view; we mutate
    pos_all[rid] = best
    set_block(reg, q_i_new, "pos", pos_all.reshape(-1))
    return q_i_new
