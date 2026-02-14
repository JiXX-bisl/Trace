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
from scripts.core.inner.blocks import BlockRegistry, decode_pos, set_block, get_block
import scripts.core.inner.coverage_metrics as coverage_metrics
import scripts.core.inner.qos_metrics as qos_metrics

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
    
    # qos connectivity
    if bool(getattr(flags, "enable_qstep_cost_qos_pos", False)):
        try: 
            total += float(
                qos_metrics.connectivity_score_from_positions(
                    cand,
                    getattr(problem, "robot_pos", np.zeros((0,2), dtype=np.float32)),
                    params
                )
            )
        except Exception:
            pass

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

    # Baseline: copy z (keeps other blocks consistent with current pipeline)
    q_i_new = np.asarray(z, dtype=np.float32).copy()
    flags = getattr(problem, "flags", None)
    params = getattr(problem, "params", None)

    # Default best position for downstream updates
    current_pos = np.asarray(problem.robot_pos[rid], dtype=np.float32).reshape(2)
    best = current_pos

    # --- Pos Update ---
    if "pos" in reg.names():
        try:
            candidates = np.asarray(problem.candidate_moves[rid], dtype=np.float32)
        except Exception:
            candidates = np.zeros((0,2), dtype=np.float32)
        
        if candidates.shape[0] > 0:
            N = int(getattr(problem, "N", 0)) or int(problem.N)
            try:
                target_all = decode_pos(
                    reg, 
                    np.asarray(z, dtype=np.float32) - np.asarray(u_i, dtype=np.float32),
                    N
                )
            except Exception:
                target_all = decode_pos(reg, np.asarray(z, dtype=np.float32), N)

            target = np.asarray(target_all[rid], dtype=np.float32).reshape(2)

            eta = float(getattr(params, "eta", 1.0)) if params is not None else 1.0

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
                    eta=eta
                )
                if best_cost is None:
                    best_cost = c
                    best_idx = idx
                else:
                    if (c < best_cost) or (c == best_cost and idx < best_idx):
                        best_cost = c
                        best_idx = idx
            best = np.asarray(candidates[best_idx], dtype=np.float32).reshape(2)
            pos_all = np.asarray(decode_pos(reg, np.asarray(z, dtype=np.float32), N), dtype=np.float32)
            pos_all = pos_all.copy()
            if pos_all.ndim == 2 and pos_all.shape[0] > rid:
                pos_all[rid] = best
                set_block(reg, q_i_new, "pos", pos_all.reshape(-1))
    
    # --- Other blocks ---
    # 1) y_hat: coverage preference
    y_hat_pref: Optional[np.ndarray] = None
    if ("y_hat" in reg.names()) and bool(getattr(flags, "enable_qstep_update_y_hat", True)):
        try:
            raw_scores = coverage_metrics.coverage_group_scores(problem, rid, best)
            y_hat_pref = coverage_metrics.normalize_to_simplex_nonneg(raw_scores)
            set_block(reg, q_i_new, "y_hat", np.asarray(y_hat_pref, dtype=np.float32))
        except Exception:
            y_hat_pref = None
    
    # 2) sigma: coupled to y_hat
    if ("sigma" in reg.names()) and bool(getattr(flags, "enable_qstep_update_sigma", True)):
        try:
            if y_hat_pref is None and ("y_hat" in reg.names()):
                y_hat_pref = np.asarray(get_block(reg, q_i_new, "y_hat"), dtype=np.float32).reshape(-1)
                y_hat_pref = coverage_metrics.normalize_to_simplex_nonneg(y_hat_pref)
            if y_hat_pref is not None:
                sigma_pref = coverage_metrics.sigma_target_from_y_hat(y_hat_pref, params, flags)
                set_block(reg, q_i_new, "sigma", np.asarray(sigma_pref, dtype=np.float32))
        except Exception:
            pass
    incident_idx: Optional[np.ndarray] = None
    cap_all: Optional[np.ndarray] = None
    B_full: Optional[np.ndarray] = None

    # 3) B_hat : QoS-aware budget allocation on incident edges
    if ("B_hat" in reg.names()) and bool(getattr(flags, "enable_qstep_update_B_hat", True)):
        try:
            link = getattr(problem, "link", None)
            edges = np.asarray(getattr(link, "edges", np.zeros((0, 2), dtype=np.int32)))
            incident_idx = qos_metrics.incident_edge_indices(edges, rid)
            E = int(getattr(problem, "E", edges.shape[0] if edges.ndim == 2 else 0))

            if incident_idx.size > 0 and E > 0:
                cap_all = np.asarray(getattr(link, "capacity", np.zeros((E,), dtype=np.float32)), dtype=np.float32).reshape(-1)
                if cap_all.size != E:
                    cap_all = np.zeros((E,), dtype=np.float32)
                elif cap_all.size == 1:
                    cap_all = np.full((E,), float(cap_all[0]), dtype=np.float32)
                else:
                    cap_all = cap_all[:E]
                    if cap_all.size < E:
                        cap_all = np.pad(cap_all, (0, E - cap_all.size))

                cap_inc = cap_all[incident_idx]
                weights = qos_metrics.edge_quality_weights(link, params, flags, rid, incident_idx)

                local_budget = getattr(params, "qstep_local_budget", None)
                if local_budget is None:
                    ratio = float(getattr(params, "qstep_local_budget_ratio", 1.0))
                    if not np.isfinite(ratio):
                        ratio = 1.0
                    local_budget = float(np.sum(cap_inc)) * float(ratio)
                alloc_inc = qos_metrics.allocate_over_incident_edges(cap_inc, weights, float(local_budget))

                B_full = np.asarray(get_block(reg, q_i_new, "B_hat"), dtype=np.float32).reshape(-1).copy()
                if B_full.size != E:
                    if B_full.size == 0:
                        B_full = np.zeros((E,), dtype=np.float32)
                    elif B_full.size == 1:
                        B_full = np.full((E,), float(B_full[0]), dtype=np.float32)
                    else:
                        B_full = B_full[:E]
                        if B_full.size < E:
                            B_full = np.pad(B_full, (0, E - B_full.size))
                B_full[incident_idx] = np.asarray(alloc_inc, dtype=np.float32)
                set_block(reg, q_i_new, "B_hat", B_full.astype(np.float32, copy=False))
        except Exception:
            pass
    
    # 4) f_hat: QoS-aware flow
    if ("f_hat" in reg.names()) and bool(getattr(flags, "enable_qstep_update_f_hat", True)):
        try:
            link = getattr(problem, "link", None)
            edges = np.asarray(getattr(link, "edges", np.zeros((0, 2), dtype=np.int)))
            E = int(getattr(problem, "E", edges.shape[0] if edges.ndim == 2 else 0))
            if incident_idx is None:
                incident_idx = qos_metrics.incident_edge_indices(edges, rid)
            if cap_all is None and E > 0:
                cap_all = np.asarray(getattr(link, "capacity", np.zeros((E,), dtype=np.float32)), dtype=np.float32).reshape(-1)
                if cap_all.size != E:
                    cap_all = np.zeros((E,), dtype=np.float32)
                elif cap_all.size == 1:
                    cap_all = np.full((E,), float(cap_all[0]), dtype=np.float32)
                else:
                    cap_all = cap_all[:E]
                    if cap_all.size < E:
                        cap_all = np.pad(cap_all, (0, E - cap_all.size))
            f_full = np.asarray(get_block(reg, q_i_new, "f_hat"), dtype=np.float32).reshape(-1).copy()
            if f_full.size != E:
                if f_full.size == 0:
                    f_full = np.zeros((E,), dtype=np.float32)
                elif f_full.size == 1:
                    f_full = np.full((E,), float(f_full[0]), dtype=np.float32)
                else:
                    f_full = f_full[:E]
                    if f_full.size < E:
                        f_full = np.pad(f_full, (0, E - f_full.size))
            if incident_idx is not None and incident_idx.size > 0 and E > 0:
                if B_full is None and ("B_hat" in reg.names()):
                    B_full = np.asarray(get_block(reg, q_i_new, "B_hat"), dtype=np.float32).reshape(-1)
                if B_full is None or B_full.size != E:
                    B_full = np.zeros((E,), dtype=np.float32)

                ratio = float(getattr(params, "qstep_f_from_B_ratio", 1.0))
                if not np.isfinite(ratio):
                    ratio = 1.0
                f_inc = np.asarray(B_full[incident_idx], dtype=np.float32) * float(ratio)
                if cap_all is not None and cap_all.size == E:
                    f_inc = np.minimum(f_inc, cap_all[incident_idx])
                f_full[incident_idx] = f_inc.astype(np.float32)
                set_block(reg, q_i_new, "f_hat", f_full.astype(np.float32, copy=False))
        except Exception:
            pass

    # 5) r_hat: QoS violation summary and coverage proxy
    if ("r_hat" in reg.names()) and bool(getattr(flags, "enable_qstep_update_r_hat", True)):
        try:
            link = getattr(problem, "link", None)
            edges = np.asarray(getattr(link, "edges", np.zeros((0, 2), dtype=np.int32)))
            E = int(getattr(problem, "E", edges.shape[0] if edges.ndim == 2 else 0))

            if B_full is None and ("B_hat" in reg.names()) and E > 0:
                B_full = np.asarray(get_block(reg, q_i_new, "B_hat"), dtype=np.float32).reshape(-1)
                if B_full.size != E:
                    B_full = np.zeros((E,), dtype=np.float32)
            if B_full is None:
                B_full = np.zeros((E,), dtype=np.float32)

            qos_scalar = float(qos_metrics.summarize_qos_violation(link, B_full, params, flags, rid))

            include_cov = bool(getattr(flags, "enable_rhat_include_coverage", True))
            cov_scalar = 0.0

            # 只有 y_hat_pref 存在时才加 coverage proxy，但 r_hat 必须始终写 qos_scalar
            if include_cov and (y_hat_pref is not None):
                yv = coverage_metrics.normalize_to_simplex_nonneg(y_hat_pref)
                G = int(yv.size)
                if G > 1:
                    cov_scalar = float((np.sum(yv * yv) - 1.0 / G) / (1.0 - 1.0 / G))
                    cov_scalar = float(np.clip(cov_scalar, 0.0, 1.0))

            scalar = float(np.clip(qos_scalar, 0.0, 1.0))
            if include_cov and (y_hat_pref is not None):
                scalar = float(np.clip(0.5 * (scalar + cov_scalar), 0.0, 1.0))

            r_min = float(getattr(params, "r_min", 0.0)) if params is not None else 0.0
            r_max = float(getattr(params, "r_max", 1.0)) if params is not None else 1.0
            lo, hi = (r_min, r_max) if r_min <= r_max else (r_max, r_min)

            r_val = lo + (hi - lo) * scalar

            r_full = np.asarray(get_block(reg, q_i_new, "r_hat"), dtype=np.float32).reshape(-1).copy()
            if r_full.size > rid:
                r_full[rid] = float(np.clip(r_val, lo, hi))
                set_block(reg, q_i_new, "r_hat", r_full.astype(np.float32, copy=False))

        except Exception:
            pass
    
    return q_i_new
