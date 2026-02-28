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

import sys
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
from scripts.core.inner.stage0_init import stage0_init_if_needed
from scripts.core.inner.diagnostics import attach_stagec_diagnostics
from scripts.core.inner import qos_metrics

# =============================================================================
# Public entry
# =============================================================================

def solve_inner_cadmm_entry(
    env: CadmmProblem,  # 在外部代码中构建好后输入
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
        raise TypeError("Build_problem_snapshot expects a CadmmProblem or your own builder wrapper.")

    prob = env_or_state

    # --- link selection path (default passthrough) ---
    link_current: LinkSnapshot = prob.link
    window_state = prob.window

    link_used = link_current
    window_new = window_state

    enable_freeze = bool(getattr(flags, "enable_link_freeze", False))
    enable_ttl = bool(getattr(flags, "enable_ttl_filter", False))

    if enable_freeze:
        W = int(getattr(window_state, "W", getattr(flags, "window_steps", 1)))
        window_new = open_or_update_window(step, link_current, window_state, W, flags, rng)
        link_used = select_window_link(step, link_current, window_new, flags)
        if bool(getattr(flags, "enable_stage0_init", True)):
            stage0_init_if_needed(step, window_new, window_new.frozen_link, params, flags)
    
    if window_new is not None:
        setattr(window_new, "staleness_strict", bool(getattr(params, "staleness_strict", False)))

    if enable_ttl:
        # link_used = apply_ttl_filter(link_used, ttl_hops=int(getattr(params, "ttl_hops", 0)),
        #                              ttl_steps=int(getattr(params, "t_fresh", 0)), flags=flags)
        ttl_steps_val = int(getattr(params, "ttl_steps", getattr(params, "t_fresh", 0)))
        link_used = apply_ttl_filter(
            link_current,
            ttl_hops=int(getattr(params, "ttl_hops", 0)),
            ttl_steps=ttl_steps_val,
            flags=flags,
            step=step,
            window_state=window_new,
            N=prob.N,
            params=params
        )

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
        coord_dim=flags.coord_dim
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

    # === 在外部初始化的时候需要完成的设置 ===
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

    # assembled ops
    enable_assembled = bool(getattr(problem.flags, "enable_assembled_ops", False))
    assembled_avg_active_only = bool(getattr(problem.flags, "assembled_avg_active_only", False))
    assembled_u_update_active_only = bool(getattr(problem.flags, "assembled_u_update_active_only", False))
    assembled_eps = float(getattr(problem.params, "assembled_eps", 1e-8))

    # qos budget closed loop
    enable_budget_cache_update = bool(getattr(problem.flags, "enable_budget_cache_update", False))
    enable_budget_soft_violation = bool(getattr(problem.flags, "enable_budget_soft_violation", False))
    budget_update_source = str(getattr(problem.flags, "budget_update_source", "z"))

    # async options
    enable_async = bool(getattr(problem.flags, "enable_async_updates", False))
    async_update_u_all = bool(getattr(problem.flags, "async_update_u_all", True))

    # residual balancing
    enable_rb = bool(getattr(problem.flags, "enable_residual_balancing", False))
    # === 在外部初始化的时候需要完成的设置 ===

    # init
    # z = np.zeros((D,), dtype=np.float32)
    # q = np.zeros((N, D), dtype=np.float32)
    # u = np.zeros((N, D), dtype=np.float32)
    # if warm_start is not None:
    #     z0 = np.asarray(warm_start.z, dtype=np.float32)
    #     q0 = np.asarray(warm_start.q, dtype=np.float32)
    #     u0 = np.asarray(warm_start.u, dtype=np.float32)
    #     if z0.shape == (D,): z = z0.copy()
    #     if q0.shape == (N, D): q = q0.copy()
    #     if u0.shape == (N, D): u = u0.copy()


    # init: prefer warm_start; otherwise initialize z from current snapshot (esp. pos)
    use_ws = False
    if warm_start is not None:
        z0 = np.asarray(warm_start.z, dtype=np.float32)
        q0 = np.asarray(warm_start.q, dtype=np.float32)
        u0 = np.asarray(warm_start.u, dtype=np.float32)
        # 重要：必须三者同时匹配才算有效 warm start
        use_ws = (z0.shape == (D,)) and (q0.shape == (N, D)) and (u0.shape == (N, D))
        if use_ws:
            z, q, u = z0.copy(), q0.copy(), u0.copy()
        

    if not use_ws:
        # 关键修复：不要用全 0 的 z/q/u；至少把 pos 初始化到当前 robot_pos
        z = np.zeros((D,), dtype=np.float32)
        u = np.zeros((N, D), dtype=np.float32)

        # --- snapshot -> z ---
        if "pos" in reg.names():
            rp = np.asarray(problem.robot_pos, dtype=np.float32)
            # ensure shape (N, coord_dim)
            if rp.ndim != 2 or rp.shape[0] != N:
                raise ValueError(f"problem.robot_pos must be (N,coord_dim), got {rp.shape}")
            if rp.shape[1] != problem.coord_dim:
                # 容错：不足则补 0，过多则截断
                if rp.shape[1] < problem.coord_dim:
                    pad = np.zeros((N, problem.coord_dim - rp.shape[1]), dtype=np.float32)
                    rp = np.concatenate([rp, pad], axis=1)
                else:
                    rp = rp[:, :problem.coord_dim]
            # pos block shape = (coord_dim*N,)
            set_block(reg, z, "pos", rp.reshape(-1))

        if "cov" in reg.names():
            set_block(reg, z, "cov", np.asarray([float(problem.coverage)], dtype=np.float32))

        if "y_hat" in reg.names():
            G = reg.shape("y_hat")[0]
            if G > 0:
                set_block(reg, z, "y_hat", np.full((G,), 1.0 / float(G), dtype=np.float32))

        # 其它块（sigma/f_hat/B_hat/r_hat）保持 0 即可

        # 每个机器人本地变量 q 初始化为 z（避免 q-step 被 0 吸到地图角落）
        q = np.tile(z[None, :], (N, 1)).astype(np.float32, copy=False)

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
        # active_mask = get_active_set(k, N, problem.flags, rng) if enable_async else np.ones((N,), dtype=bool)
        hint = getattr(problem.window, "active_hint", None)
        active_mask = (
            get_active_set(k, N, problem.flags, rng, active_hint=hint) if enable_async else np.ones((N,), dtype=bool)
        )
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
            q_plus_u = q_hat + u
        else:
            q_hat = None
            q_plus_u = q + u

        if enable_assembled:
            from scripts.core.inner.assembled_ops import owner_mask_for_block, assembled_average

            delta = getattr(problem.window, "active_hint", None) if problem.window is not None else None
            if assembled_avg_active_only and delta is not None:
                active_avg = np.asarray(delta, dtype=bool).reshape(-1)
                if active_avg.shape != (N,):
                    active_avg = np.ones((N,), dtype=bool)
            else:
                active_avg = np.ones((N,), dtype=bool)

            z_tilde = np.zeros((D,), dtype=np.float32)
            for name in reg.names():
                sl = reg.sl(name)
                dim = int(reg.size(name)) if hasattr(reg, "size") else int(sl.stop - sl.start)
                blk_vals = np.asarray(q_plus_u[:, sl], dtype=np.float32).reshape(N, dim)
                own = owner_mask_for_block(name, problem, reg, problem.flags)
                z_blk = assembled_average(name, blk_vals, own, active_avg, assembled_eps)
                z_tilde[sl] = np.asarray(z_blk, dtype=np.float32).reshape(-1)
        else:
            z_tilde = np.mean(q_plus_u, axis=0)

        z_tilde_blocks = reg.unpack(z_tilde)
        constraints_by_block = build_constraints(problem, reg, z_blocks=z_tilde_blocks)
        z_proj_blocks, diag_by_block = project_blocks(
            z_tilde_blocks, constraints_by_block,
            method=proj_method, iters=proj_iters, tol=proj_tol
        )

        # Scheme-2 coupled feasibility passes (cross-block constraints affecting z)
        from scripts.core.inner.coupled_projection_passes import apply_coupled_projection_passes

        apply_coupled_projection_passes(
            problem,
            reg,
            z_proj_blocks,
            diag_by_block,
            method=proj_method,
            iters=proj_iters,
            tol=proj_tol,
        )

        z = reg.pack(z_proj_blocks)

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
        if assembled_u_update_active_only:
            delta_u = getattr(problem.window, "active_hint", None) if problem.window is not None else None
            if delta_u is None:
                u_mask = active_mask
            else:
                u_mask = np.asarray(delta_u, dtype=bool).reshape(-1)
        else:
            u_mask = active_mask
        
        if async_update_u_all:
            u = u + (q - z[None, :])
        else:
            u = u_prev.copy()
            u[u_mask] = u_prev[u_mask] + (q[u_mask] - z[None, :])
        
        # --- QoS budget closed loop ---
        if (problem.window is not None) and (enable_budget_cache_update or enable_budget_soft_violation):
            src = z if budget_update_source == "z" else q
            qos_metrics.update_budget_cache(
                window_state=problem.window,
                reg=reg,
                value=src,
                ema=float(getattr(problem.params, "budget_cache_ema", 0.0)),
                enable_update=enable_budget_cache_update,
                enable_soft_violation=enable_budget_soft_violation,
                tol=float(getattr(problem.params, "budget_violation_tol", 0.0)),
                flags=problem.flags,
            )
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

    next_pos = decode_pos(reg, z, problem.N, problem.coord_dim) if "pos" in reg.names() else np.asarray(problem.robot_pos, dtype=np.float32)

    sol = InnerSolution(next_pos=next_pos, z=z, q=q, u=u)
    ws = CadmmWarmStart(z=z.copy(), u=u.copy(), q=q.copy())
    diag = CadmmDiagnostics(
        iters=iters_used,
        r_norm=r_blk_last,
        s_norm=s_blk_last,
        eta_hist=eta_hist,
        proj_violation=proj_violation,
    )
    attach_stagec_diagnostics(problem, reg, diag)

    # attach optional extra diagnostics (doesn't change dataclass definition)
    if enable_async:
        setattr(diag, "async_active_hist", async_active_hist)
    if enable_rb:
        setattr(diag, "residual_balance_events", rb_events)

    # window info is better recorded at entry level (per env step), but we keep hook
    return sol, ws, diag

