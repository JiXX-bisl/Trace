"""
scripts.core.cadmm_solver

TRACE / Inner C-ADMM *entry orchestrator*.

Layering
--------
- blocks.py      : variable protocol (BlockRegistry), residual protocol, eps
- constraints.py : feasible-set description (per-block constraint lists)
- projections.py : projection computations for z-step
- cadmm_solver.py: orchestrates q-step / z-step / u-step / stopping / diagnostics

This file MUST NOT depend on concrete env/robot/map classes. Any such logic
should live in a builder function that returns numpy snapshots only.

STRICT I/O PROTOCOL
-------------------
D = problem.reg.total_dim
- z: (D,)
- q,u: (N,D)
All block access must go through BlockRegistry helpers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
from dataclasses import replace

import numpy as np

from scripts.core.data import (
    CadmmProblem,
    CadmmWarmStart,
    InnerSolution,
    CadmmDiagnostics,
    FeatureFlags,
    LinkSnapshot,
    TaskSnapshot,
    CommWindowState,
)
from scripts.core.inner.blocks import (
    BlockRegistry,
    decode_pos,
    make_registry,
    set_block,
    make_residual_model,
    compute_eps_pri,
    compute_eps_dual,
)
from scripts.core.inner.constraints import build_constraints
from scripts.core.inner.projections import project_blocks, project_one_block

from scripts.core.inner.window import open_or_update_window, select_window_link, apply_ttl_filter
from scripts.core.inner.async_scheduler import get_active_set
from scripts.core.inner.residual_balancing import residual_balance_step
from scripts.core.inner.q_step import solve_local_q

# =============================================================================
# Public entry
# =============================================================================

def solve_inner_cadmm_entry(
    env: Any,
    step: int,
    params: Any,
    flags: FeatureFlags,
    warm_start: Optional[CadmmWarmStart],
    rng: np.random.Generator,
    optional_overrides: Optional[Dict[str, Any]] = None
) -> Tuple[InnerSolution, CadmmWarmStart, CadmmDiagnostics]:
    """
    Stable entry point for the Inner C-ADMM solver.

    env can be:
    - a CadmmProblem (unit tests / offline)
    - a dict-like or attribute-like object with snapshot fields

    Returns:
        (InnerSolution, CadmmWarmStart, CadmmDiagnostics)
    """
    problem = build_problem_snapshot(env, step, params, flags, rng, optional_overrides)
    return solve_inner_cadmm(problem, warm_start, rng)


# =============================================================================
# Snapshot builder
# =============================================================================

def build_problem_snapshot(
    env_or_state: Any,
    step: int,
    params: Any,
    flags: FeatureFlags,
    rng: np.random.Generator,
    optional_overrides: Optional[Dict[str, Any]] = None,
) -> CadmmProblem:
    """
    Build frozen CadmmProblem snapshot.

    Integration:
      - window freeze (enable_link_freeze)
      - ttl filter     (enable_ttl_filter)
    Both are optional; default keeps Step4 behavior.
    """
    if not isinstance(env_or_state, CadmmProblem):
        raise TypeError("Step5: build_problem_snapshot expects a CadmmProblem or your own builder wrapper.")

    prob = env_or_state

    # --- link selection path (default passthrough) ---
    link_current: LinkSnapshot = prob.link
    window_state = prob.window

    link_used = link_current
    window_new = window_state

    enable_freeze = bool(getattr(flags, "enable_link_freeze", False))
    enable_ttl = bool(getattr(flags, "enable_ttl_filter", False))

    if enable_freeze:
        W = int(getattr(window_state, "W", 1))
        window_new = open_or_update_window(step, link_current, window_state, W, flags, rng)
        link_used = select_window_link(step, link_current, window_new, flags)

    if enable_ttl:
        link_used = apply_ttl_filter(link_used, ttl_hops=int(getattr(params, "ttl_hops", 0)),
                                     ttl_steps=int(getattr(params, "t_fresh", 0)), flags=flags)

    E_used = int(np.asarray(link_used.edges).shape[0])
    flags_used = flags
    if E_used == 0:
        if getattr(flags, "enable_f_hat", False) or getattr(flags, "enable_B_hat", False):
            flags_used = replace(
                flags,
                enable_f_hat=False,
                enable_B_hat=False,
                enable_proj_f_hat=False,
                enable_proj_B_hat=False,
            )

    # return frozen snapshot (E may change after ttl filter)
    return CadmmProblem(
        N=prob.N,
        E=E_used,
        G=prob.G,
        T=prob.T,
        robot_pos=np.asarray(prob.robot_pos, dtype=np.float32),
        candidate_moves=list(prob.candidate_moves),
        coverage=float(prob.coverage),
        frontier_entropy=np.asarray(prob.frontier_entropy, dtype=np.float32),
        repulsion_grad=np.asarray(prob.repulsion_grad, dtype=np.float32),
        link=link_used,
        task=prob.task,
        params=params,
        flags=flags_used,
        window=window_new,
    )

# =============================================================================
# Inner C-ADMM core
# =============================================================================

def solve_inner_cadmm(
    problem: CadmmProblem,
    warm_start: Optional[CadmmWarmStart] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[InnerSolution, CadmmWarmStart, CadmmDiagnostics]:
    if rng is None:
        rng = np.random.default_rng()

    reg = make_registry(problem.N, problem.E, problem.G, problem.flags)
    D = reg.total_dim
    N = problem.N

    eta = float(getattr(problem.params, "eta", 1.0))
    alpha = float(getattr(problem.params, "alpha", 1.0))
    k_max = int(getattr(problem.params, "k_max", 30))

    eps_pri_abs = float(getattr(problem.params, "eps_pri", 1e-3))
    eps_dual_abs = float(getattr(problem.params, "eps_dual", 1e-3))
    eps_rel = 0.0  # compat with your params fields

    # projection options
    proj_method = str(getattr(problem.flags, "proj_method", "dykstra"))
    proj_iters = int(getattr(problem.flags, "proj_iters", 80))
    proj_tol = float(getattr(problem.flags, "proj_tol", 1e-6))
    enable_over_relax = bool(getattr(problem.flags, "enable_over_relax", False))

    # async options
    enable_async = bool(getattr(problem.flags, "enable_async_updates", False))
    async_update_u_all = bool(getattr(problem.flags, "async_update_u_all", True))

    # residual balancing
    enable_rb = bool(getattr(problem.flags, "enable_residual_balancing", False))

    # init
    z = np.zeros((D,), dtype=np.float32)
    q = np.zeros((N, D), dtype=np.float32)
    u = np.zeros((N, D), dtype=np.float32)

    if warm_start is not None:
        z0 = np.asarray(warm_start.z, dtype=np.float32)
        q0 = np.asarray(warm_start.q, dtype=np.float32)
        u0 = np.asarray(warm_start.u, dtype=np.float32)
        if z0.shape == (D,): z = z0.copy()
        if q0.shape == (N, D): q = q0.copy()
        if u0.shape == (N, D): u = u0.copy()

    res_model = make_residual_model(problem, reg, problem.flags)

    eta_hist: list[float] = []
    proj_violation: Dict[str, float] = {name: 0.0 for name in reg.names()}
    r_blk_last: Dict[str, float] = {name: 0.0 for name in reg.names()}
    s_blk_last: Dict[str, float] = {name: 0.0 for name in reg.names()}

    # Optional diagnostics (not in dataclass): async masks / window info / rb events
    async_active_hist: list[np.ndarray] = []
    rb_events: list[dict] = []

    iters_used = 0

    for k in range(k_max):
        iters_used = k + 1
        eta_hist.append(float(eta))
        z_prev = z.copy()
        q_prev = q.copy()
        u_prev = u.copy()

        # --- async mask ---
        active_mask = get_active_set(k, N, problem.flags, rng) if enable_async else np.ones((N,), dtype=bool)
        if enable_async:
            async_active_hist.append(active_mask.copy())

        # --- q-step ---
        q_new = q_prev.copy()
        for i in range(N):
            if active_mask[i]:
                q_new[i] = solve_local_q(
                    rid=i, problem=problem, reg=reg,
                    z=z, u_i=u[i], q_i_prev=q_prev[i],
                    rng=rng,
                )
            else:
                q_new[i] = q_prev[i]
        q = q_new

        # --- z-step ---
        if enable_over_relax:
            q_hat = alpha * q + (1.0 - alpha) * z_prev[None, :]
            z_tilde = np.mean(q_hat + u, axis=0)
        else:
            z_tilde = np.mean(q + u, axis=0)

        z_tilde_blocks = reg.unpack(z_tilde)
        constraints_by_block = build_constraints(problem, reg, z_blocks=z_tilde_blocks)
        z_proj_blocks, diag_by_block = project_blocks(
            z_tilde_blocks, constraints_by_block,
            method=proj_method, iters=proj_iters, tol=proj_tol
        )

        # strict sigma coupling based on projected y_hat
        if (
            bool(getattr(problem.flags, "enable_sigma_coupled_to_y", True)) 
            and ("sigma" in z_proj_blocks)
            and ("y_hat" in z_proj_blocks)
        ):
            try:
                cons2 = build_constraints(problem=problem, reg=reg, z_blocks=z_proj_blocks)
                sigma_cons = cons2.get("sigma", [])
                sigma_new, sigma_info2 = project_one_block(
                    z_proj_blocks["sigma"],
                    sigma_cons,
                    method=proj_method,
                    iters=proj_iters,
                    tol=proj_tol
                )
                z_proj_blocks["sigma"] = sigma_new
                d0 = diag_by_block.get(
                    "sigma",
                    {"iters_used": 0, "violations": [], "max_violation": 0.0, "delta_norm": 0.0}
                )
                d0 = dict(d0)
                d0["coupled_pass"] = sigma_info2
                d0["max_violation"] = float(
                    max(
                        float(d0.get("max_violation", 0.0) or 0.0),
                        float(sigma_info2.get("max_violation", 0.0) or 0.0),
                    )
                )
                diag_by_block["sigma"] = d0
            except Exception:
                pass

        z = reg.pack(z_proj_blocks)

        proj_violation = {
            name: float(diag_by_block.get(name, {}).get("max_violation", 0.0))
            for name in reg.names()
        }

        # --- u-step ---
        if async_update_u_all:
            u = u + (q - z[None, :])
        else:
            u = u_prev.copy()
            u[active_mask] = u_prev[active_mask] + (q[active_mask] - z[None, :])

        # --- residuals ---
        r_blk, r_global = res_model.primal_block_norms(q, z, reg)
        s_blk, s_global = res_model.dual_block_norms(z, z_prev, eta, reg)
        r_blk_last = {kk: float(vv) for kk, vv in r_blk.items()}
        s_blk_last = {kk: float(vv) for kk, vv in s_blk.items()}

        eps_pri_val = float(compute_eps_pri(eps_pri_abs, eps_rel, q, z))
        eps_dual_val = float(compute_eps_dual(eps_dual_abs, eps_rel, u, eta))

        # --- stop check (based on current iteration state) ---
        if (r_global <= eps_pri_val) and (s_global <= eps_dual_val):
            break

        # --- residual balancing for NEXT iteration (optional) ---
        if enable_rb:
            eta_new, u_scaled, info = residual_balance_step(eta, u, r_global, s_global, problem.params, problem.flags)
            if info.get("changed", False):
                rb_events.append(info)
            eta = float(eta_new)
            u = np.asarray(u_scaled, dtype=np.float32)

    next_pos = decode_pos(reg, z, problem.N) if "pos" in reg.names() else np.asarray(problem.robot_pos, dtype=np.float32)

    sol = InnerSolution(next_pos=next_pos, z=z, q=q, u=u)
    ws = CadmmWarmStart(z=z.copy(), u=u.copy(), q=q.copy())
    diag = CadmmDiagnostics(
        iters=iters_used,
        r_norm=r_blk_last,
        s_norm=s_blk_last,
        eta_hist=eta_hist,
        proj_violation=proj_violation,
    )

    # attach optional extra diagnostics (doesn't change dataclass definition)
    if enable_async:
        setattr(diag, "async_active_hist", async_active_hist)
    if enable_rb:
        setattr(diag, "residual_balance_events", rb_events)

    # window info is better recorded at entry level (per env step), but we keep hook
    return sol, ws, diag


# # =============================================================================
# # q-step
# # =============================================================================

# def solve_local_q(
#     rid: int,
#     problem: CadmmProblem,
#     reg,
#     z: np.ndarray,
#     u_i: np.ndarray,
#     q_i_prev: np.ndarray,
#     rng: np.random.Generator,
# ) -> np.ndarray:
#     """
#     Minimal q-step: only updates pos using discrete projection to (z - u_i).pos.
#     Other blocks copy z (keeps dimensions consistent, z-step handles constraints).
#     """
#     q_i = np.asarray(z, dtype=np.float32).copy()

#     if "pos" not in reg.names():
#         return q_i

#     pos_target_all = decode_pos(reg, z - u_i, problem.N)
#     target = pos_target_all[rid]

#     candidates = np.asarray(problem.candidate_moves[rid], dtype=np.float32).reshape(-1, 2)
#     if candidates.shape[0] == 0:
#         return q_i

#     d2 = np.sum((candidates - target[None, :]) ** 2, axis=1)
#     best = candidates[int(np.argmin(d2))]

#     pos_all = decode_pos(reg, z, problem.N)
#     pos_all[rid] = best
#     set_block(reg, q_i, "pos", pos_all.reshape(-1))
#     return q_i
