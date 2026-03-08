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
    CadmmLog,
    CadmmLogConfig,
    CadmmRunLog,
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
    eq_mask
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
        K=getattr(prob, "K", None),
        T_horizon=int(getattr(prob, "T_horizon", 1)),
        robot_pos=np.asarray(prob.robot_pos, dtype=np.float32),
        candidate_moves=list(prob.candidate_moves),
        coverage=float(prob.coverage),
        frontier_entropy=np.asarray(prob.frontier_entropy, dtype=np.float32),
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
    reg = make_registry(problem.N, problem.E, problem.G, problem.flags, T_horizon=int(getattr(problem, "T_horizon", 1)))
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
    theory_mode = bool(getattr(problem.flags, "enable_theory_mode", False))
    joint_poly = bool(getattr(problem.flags, "enable_sigma_y_joint_polytope", False))
    eqm = eq_mask(reg) if theory_mode else None
    if theory_mode and (eqm is not None) and (int(np.sum(eqm)) <= 0):
        # Degenerate: if eqm is empty, fall back to full dimension to avoid NaNs.
        eqm = None


    # ------------------------------------------------------------------
    # Strict theory alignment: Y average-assembly mode (mean(s_hat) - y_hat = 0)
    # This mode introduces a GLOBAL dual uY (dy,) and forbids using per-agent u[:,y_hat].
    # Hard constraints enforced here:
    #   (A) y_hat must NOT be dual-updated twice => remove from eqm + force u[:,y_hat]=0
    #   (B) stop/eps for Y must use rY=||mean_s-y|| and sY=eta_y*||Δy||/sqrt(N), NOT q_y-z_y
    #   (C) sum_s must be computed ONLY from cached q_prev.s_hat (owned rows), no online recompute
    # ------------------------------------------------------------------
    enable_y_avg_assembly = bool(getattr(problem.flags, "enable_y_avg_assembly", False))
    enable_theta_flag = bool(getattr(problem.flags, "enable_theta", False))
    YAVG = bool(theory_mode and enable_theta_flag and enable_y_avg_assembly and ("y_hat" in reg.names()) and ("s_hat" in reg.names()))
    if YAVG:
        sl_y = reg.sl("y_hat")
        dy = int(np.prod(reg.shape("y_hat")))
        if dy <= 0:
            raise ValueError("enable_y_avg_assembly=True but y_hat dim dy<=0")
        sl_s = reg.sl("s_hat")  # shape (N, dy) in each q_i flat vector
        # (A) ensure y_hat is excluded from the legacy equality mask
        if eqm is None:
            eqm = np.ones((D,), dtype=bool)
        else:
            eqm = np.asarray(eqm, dtype=bool).copy()
        eqm[sl_y] = False

    # Keep an immutable base penalty for Y (do NOT let residual balancing change it)
    eta0 = float(getattr(problem.params, "eta", 1.0))
    if (not np.isfinite(eta0)) or eta0 <= 1e-9:
        eta0 = 1.0
    eta_y = float(getattr(problem.params, "eta_y_avg_scale", 1.0)) * eta0  # used only when YAVG
    if not np.isfinite(eta_y) or eta_y <= 0.0:
        eta_y = eta0
    if YAVG:
        # For diagnostics consistency: linear_assembler.apply_dual("Y") should use
        # the same fixed eta_y as the solver stop criterion (eta-balancing must not change it).
        setattr(problem, "_eta_y_fixed", float(eta_y))

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

    # (6.1) Global dual uY for Y-average assembly (dy,)
    # Use getattr to stay compatible with older CadmmWarmStart dataclass (no field uY).
    uY = None
    if warm_start is not None:
        uY = getattr(warm_start, "uY", None)
    if YAVG:
        if uY is None:
            uY = np.zeros((dy,), dtype=np.float32)
        else:
            uY = np.asarray(uY, dtype=np.float32).reshape(-1)
            if uY.size != dy:
                raise ValueError(f"warm_start.uY dim {uY.size} != dy {dy}")

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
            # G = reg.shape("y_hat")[0]
            # if G > 0:
            #     set_block(reg, z, "y_hat", np.full((G,), 1.0 / float(G), dtype=np.float32))
            dy = int(np.prod(reg.shape("y_hat")))
            if dy > 0:
                set_block(reg, z, "y_hat", np.full((dy,), 1.0 / float(dy), dtype=np.float32))

        if "theta" in reg.names():
            shp = reg.shape("theta")      # (N, M)
            M = int(shp[-1])
            if M > 0:
                th = np.full((N, M), 1.0 / float(M), dtype=np.float32)
                # set_block(reg, z, "theta", th.reshape(-1))
                set_block(reg, z, "theta", th.reshape(reg.shape("theta")))
        # 其它块（sigma/f_hat/B_hat/r_hat）保持 0 即可

        # 每个机器人本地变量 q 初始化为 z（避免 q-step 被 0 吸到地图角落）
        q = np.tile(z[None, :], (N, 1)).astype(np.float32, copy=False)
    # ------------------------------------------------------------------
    # Route-B/theory: keep a per-solve warm-start reference for q-step priors.
    # Meaning: theta prior should come from previous SNAPSHOT (warm_start),
    # not from z or from the previous inner-iteration.
    # For legacy runs (theta disabled), this has no effect.
    # ------------------------------------------------------------------
    q_ws = q.copy()
    res_model = make_residual_model(problem, reg, problem.flags)


    # Helper: extract "owned rows" from a stacked-by-robot block in q-matrix.
    # This matches your pos extraction logic: each robot i owns row i in the stacked block.
    def _extract_owned_rows(qmat: np.ndarray, sl: slice, row_dim: int) -> np.ndarray:
        out = np.zeros((N, row_dim), dtype=np.float32)
        for ii in range(N):
            blk = np.asarray(qmat[ii, sl], dtype=np.float32).reshape(N, row_dim)
            out[ii] = blk[ii]
        return out

    # --- optional per-solve logger (rollout-level aggregation) ---
    run_log = None
    cadmm_log = getattr(problem, "log", None)
    if cadmm_log is not None:
        try:
            # start run after z/q/u are initialized (so z0/q0/u0 are meaningful)
            run_log = cadmm_log.start_run(problem=problem, reg=reg, z0=z, q0=q, u0=u)
            if run_log is not None:
                extra = []
                if theory_mode:
                    extra = ["S", "C", "Y", "R"]
                for nm in extra:
                    if nm not in run_log.block_order:
                        run_log.block_order.append(nm)
        except Exception:
            run_log = None



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

        # (A) Hard rule: in YAVG, y_hat must NOT be dual-updated by legacy u[:,y_hat].
        # Always keep u[:,y_hat]==0 to prevent mixed-dual feedback into theta-QP.
        if YAVG:
            u[:, sl_y] = 0.0

        # --- async mask ---
        # active_mask = get_active_set(k, N, problem.flags, rng) if enable_async else np.ones((N,), dtype=bool)
        hint = getattr(problem.window, "active_hint", None)
        active_mask = (
            get_active_set(k, N, problem.flags, rng, active_hint=hint) if enable_async else np.ones((N,), dtype=bool)
        )
        if enable_async:
            async_active_hist.append(active_mask.copy())
        # --- theta debug: clear per-iteration buffer (filled inside q_step) ---
        if bool(getattr(problem.flags, "enable_theta", False)):
            setattr(problem, "_theta_dbg", {})

        # (C) Hard rule: sum_s must come ONLY from cached q_prev.s_hat (owned rows).
        # Do NOT recompute Phi@theta online here (keeps async/TTL perturbation interpretable).
        if YAVG:
            s_rows_prev = _extract_owned_rows(q_prev, sl_s, dy)      # (N,dy)
            sum_s_prev = np.sum(s_rows_prev, axis=0)                # (dy,)
            # Current global y_hat (z variable) for this iteration's q-step
            z_y = np.asarray(z[sl_y], dtype=np.float32).reshape(-1)  # (dy,)


        # --- q-step ---
        q_new = q_prev.copy()
        # Route-B: theta-QP uses q_i_prev.theta as a stabilizer (prior).
        # We want this prior to come from previous snapshot warm_start, i.e., q_ws.
        theta_qp_enabled = (
            theory_mode
            and bool(getattr(problem.flags, "enable_theta", False))
            and ("theta" in reg.names())
            and bool(getattr(problem.flags, "enable_qstep_update_theta", True))
        )
        for i in range(N):
            if active_mask[i]:
                q_prev_for_qstep = q_ws[i] if theta_qp_enabled else q_prev[i]
                # (6.4) per-agent y_tgt_override for YAVG:
                # y_tgt_i = N*(z_y - uY) - S_other, where S_other = sum_s_prev - s_i_prev
                if YAVG:
                    s_i_prev = s_rows_prev[i]
                    S_other = sum_s_prev - s_i_prev
                    y_tgt_i = (float(N) * (z_y - uY) - S_other).astype(np.float32, copy=False)

                q_new[i] = solve_local_q(
                    rid=i, problem=problem, reg=reg,
                    # z=z, u_i=u[i], q_i_prev=q_prev[i],
                    z=z, u_i=u[i], q_i_prev=q_prev_for_qstep,
                    rng=rng,
                    y_tgt_override=(y_tgt_i if YAVG else None),
                )
            else:
                q_new[i] = q_prev[i]
        q = q_new
        # After q-step: compute mean_s from UPDATED q (owned rows), used by z-step and uY update.
        if YAVG:
            s_rows = _extract_owned_rows(q, sl_s, dy)                 # (N,dy)
            mean_s = np.mean(s_rows, axis=0).astype(np.float32)       # (dy,)
        # --- z-step ---
        if enable_over_relax:
            q_hat = alpha * q + (1.0 - alpha) * z_prev[None, :]
            q_plus_u = q_hat + u
        else:
            q_hat = None
            q_plus_u = q + u
        macro_keep = {"f_hat", "B_hat", "y_hat", "r_hat", "sigma"}
        if enable_assembled:
            # Theory-aligned Z_macro freeze set:
            #   - z_eq blocks: f_hat/B_hat/y_hat/r_hat
            #   - coupled block: sigma (does NOT enter equality consensus/u-step)
            
            from scripts.core.inner.assembled_ops import owner_mask_for_block, assembled_average

            delta = getattr(problem.window, "active_hint", None) if problem.window is not None else None
            if assembled_avg_active_only and delta is not None:
                active_avg = np.asarray(delta, dtype=bool).reshape(-1)
                if active_avg.shape != (N,):
                    active_avg = np.ones((N,), dtype=bool)
            else:
                active_avg = np.ones((N,), dtype=bool)

            # z_tilde = np.zeros((D,), dtype=np.float32)
            # Start from previous z so that non-macro blocks can be frozen safely.
            z_tilde = z_prev.copy().astype(np.float32, copy=False)
            for name in reg.names():
                if theory_mode and (name not in macro_keep):
                    continue
                sl = reg.sl(name)
                dim = int(reg.size(name)) if hasattr(reg, "size") else int(sl.stop - sl.start)
                blk_vals = np.asarray(q_plus_u[:, sl], dtype=np.float32).reshape(N, dim)
                own = owner_mask_for_block(name, problem, reg, problem.flags)
                z_blk = assembled_average(name, blk_vals, own, active_avg, assembled_eps)
                z_tilde[sl] = np.asarray(z_blk, dtype=np.float32).reshape(-1)
        else:
            if theory_mode:
                # Conservative macro-only center construction (B=-I on macro blocks).
                z_tilde = z_prev.copy().astype(np.float32, copy=False)
                for name in reg.names():
                    if name not in macro_keep:
                        continue
                    sl = reg.sl(name)
                    z_tilde[sl] = np.mean(np.asarray(q_plus_u[:, sl], dtype=np.float32), axis=0)
            else:
                z_tilde = np.mean(q_plus_u, axis=0)

        # (6.5) YAVG y_hat center override:
        # y_tilde = mean_s + uY (scaled dual form), NOT mean(q_y+u_y).
        if YAVG:
            z_tilde[sl_y] = (mean_s + uY).astype(np.float32, copy=False)

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

        # z = reg.pack(z_proj_blocks)

        # # strict sigma coupling based on projected y_hat
        # if (
        #     bool(getattr(problem.flags, "enable_sigma_coupled_to_y", True)) 
        #     and ("sigma" in z_proj_blocks)
        #     and ("y_hat" in z_proj_blocks)
        # ):
        #     try:
        #         cons2 = build_constraints(problem=problem, reg=reg, z_blocks=z_proj_blocks)
        #         sigma_cons = cons2.get("sigma", [])
        #         sigma_new, sigma_info2 = project_one_block(
        #             z_proj_blocks["sigma"],
        #             sigma_cons,
        #             method=proj_method,
        #             iters=proj_iters,
        #             tol=proj_tol
        #         )
        #         z_proj_blocks["sigma"] = sigma_new
        #         d0 = diag_by_block.get(
        #             "sigma",
        #             {"iters_used": 0, "violations": [], "max_violation": 0.0, "delta_norm": 0.0}
        #         )
        #         d0 = dict(d0)
        #         d0["coupled_pass"] = sigma_info2
        #         d0["max_violation"] = float(
        #             max(
        #                 float(d0.get("max_violation", 0.0) or 0.0),
        #                 float(sigma_info2.get("max_violation", 0.0) or 0.0),
        #             )
        #         )
        #         diag_by_block["sigma"] = d0
        #     except Exception:
        #         pass

        # z = reg.pack(z_proj_blocks)
        # Theory mode: z-step is projection onto Z_macro; freeze non-macro blocks.
        if theory_mode:
            z_prev_blocks = reg.unpack(z_prev)
            for name in reg.names():
                if name not in macro_keep:
                    z_proj_blocks[name] = z_prev_blocks[name]

        # Legacy sigma-only reproject (REMOVE under theory/joint polytope):
        # The joint (y_hat, sigma) coupling is already handled in apply_coupled_projection_passes.
        if not (theory_mode or joint_poly):
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
        # (6.6) Update GLOBAL dual uY after y_hat has been updated by z-step projection:
        # uY <- uY + (mean_s - y_hat)
        if YAVG:
            y_hat_now = np.asarray(z[sl_y], dtype=np.float32).reshape(-1)
            uY = (uY + (mean_s - y_hat_now)).astype(np.float32, copy=False)

        # --- u-step ---
        if assembled_u_update_active_only:
            delta_u = getattr(problem.window, "active_hint", None) if problem.window is not None else None
            if delta_u is None:
                u_mask = active_mask
            else:
                u_mask = np.asarray(delta_u, dtype=bool).reshape(-1)
        else:
            u_mask = active_mask
        
        # if async_update_u_all:
        #     u = u + (q - z[None, :])
        # else:
        #     u = u_prev.copy()
        # Theory: u-step only for z_eq entries (sigma does not enter equality assembly).
        if theory_mode and (eqm is not None):
            delta_dual = (q - z[None, :]).astype(np.float32, copy=False)
            delta_dual[:, ~eqm] = 0.0
        else:
            delta_dual = (q - z[None, :]).astype(np.float32, copy=False)

        if async_update_u_all:
            u = u + delta_dual
        else:
            u = u_prev.copy()
            # u[u_mask] = u_prev[u_mask] + (q[u_mask] - z[None, :])
            u[u_mask] = u_prev[u_mask] + delta_dual[u_mask]

        # (A) Hard rule re-enforced: keep legacy u[:,y_hat]==0 in YAVG
        if YAVG:
            u[:, sl_y] = 0.0

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

        # eps_pri_val = float(compute_eps_pri(eps_pri_abs, eps_rel, q, z))
        # eps_dual_val = float(compute_eps_dual(eps_dual_abs, eps_rel, u, eta))
        # Stop/eps are evaluated on z_eq dimension in theory mode.
        if theory_mode and (eqm is not None):
            q_eps = np.asarray(q[:, eqm], dtype=np.float32)
            z_eps = np.asarray(z[eqm], dtype=np.float32)
            u_eps = np.asarray(u[:, eqm], dtype=np.float32)
            eps_pri_val = float(compute_eps_pri(eps_pri_abs, eps_rel, q_eps, z_eps))
            eps_dual_val = float(compute_eps_dual(eps_dual_abs, eps_rel, u_eps, eta))
            # r_stop = float(np.linalg.norm((q_eps - z_eps[None, :]).reshape(-1), ord=2))
            # s_stop = float(np.linalg.norm((eta * (z_eps - np.asarray(z_prev[eqm], dtype=np.float32))).reshape(-1), ord=2))
            r_eq = float(np.linalg.norm((q_eps - z_eps[None, :]).reshape(-1), ord=2))
            s_eq = float(np.linalg.norm((eta * (z_eps - np.asarray(z_prev[eqm], dtype=np.float32))).reshape(-1), ord=2))
            r_stop = r_eq
            s_stop = s_eq            
        else:
            eps_pri_val = float(compute_eps_pri(eps_pri_abs, eps_rel, q, z))
            eps_dual_val = float(compute_eps_dual(eps_dual_abs, eps_rel, u, eta))
            r_stop = float(r_global)
            s_stop = float(s_global)        

        # (6.7) Hard rule: stop/eps for YAVG must use the NEW Y residual definition.
        if YAVG:
            # primal Y residual: rY = ||mean_s - y||
            y_hat_now = np.asarray(z[sl_y], dtype=np.float32).reshape(-1)
            y_hat_prev = np.asarray(z_prev[sl_y], dtype=np.float32).reshape(-1)
            rY = float(np.linalg.norm((mean_s - y_hat_now).reshape(-1), ord=2))

            # dual Y residual (scaled, consistent with A/B for mean constraint):
            # sY = eta_y * ||Δy|| / sqrt(N)
            dy_s = float(np.linalg.norm((y_hat_now - y_hat_prev).reshape(-1), ord=2))
            sY = float(eta_y * dy_s / np.sqrt(float(N)))

            # eps for Y part (use same abs, rel policy)
            eps_pri_Y = float(np.sqrt(float(dy)) * eps_pri_abs + eps_rel * max(
                float(np.linalg.norm(mean_s, ord=2)), float(np.linalg.norm(y_hat_now, ord=2))
            ))
            eps_dual_Y = float(np.sqrt(float(dy)) * eps_dual_abs + eps_rel * float(
                eta_y * np.linalg.norm(uY, ord=2) / np.sqrt(float(N))
            ))

            # combine eq + Y for global stop
            r_stop = float(np.sqrt(float(r_stop * r_stop + rY * rY)))
            s_stop = float(np.sqrt(float(s_stop * s_stop + sY * sY)))
            eps_pri_val = float(np.sqrt(float(eps_pri_val * eps_pri_val + eps_pri_Y * eps_pri_Y)))
            eps_dual_val = float(np.sqrt(float(eps_dual_val * eps_dual_val + eps_dual_Y * eps_dual_Y)))       

        # --- stop check (based on current iteration state) ---
        # if (r_global <= eps_pri_val) and (s_global <= eps_dual_val):
        
        # --- optional logging (per-iter) ---
        if run_log is not None:
            try:
                cfg = getattr(cadmm_log, "config", None)
                stride = int(getattr(cfg, "stride", 1) or 1) if cfg is not None else 1
                if stride < 1:
                    stride = 1
                if (int(k) % stride) == 0:
                    store_blocks = tuple(getattr(cfg, "store_blocks", ("pos","f_hat","B_hat","y_hat","sigma","r_hat"))) if cfg is not None else ("pos","f_hat","B_hat","y_hat","sigma","r_hat")
                    store_full_state = bool(getattr(cfg, "store_full_state", False) or (str(getattr(cfg, "level", "lite")).lower() == "full")) if cfg is not None else False
                    state_dtype = str(getattr(cfg, "state_dtype", "float32")) if cfg is not None else "float32"

                    # coupled pass summary (small & stable)
                    coupled_summary: Dict[str, Any] = {}
                    for nm in ("y_hat", "sigma", "f_hat", "B_hat", "r_hat"):
                        cp = diag_by_block.get(nm, {}).get("coupled_pass", None)
                        if isinstance(cp, dict) and cp:
                            coupled_summary[nm] = {
                                "op": cp.get("op", None),
                                "method": cp.get("method", None),
                                "pre_max_violation": float(cp.get("pre_max_violation", 0.0) or 0.0),
                                "max_violation": float(cp.get("max_violation", 0.0) or 0.0),
                                "delta_norm": float(cp.get("delta_norm", 0.0) or 0.0),
                            }
                            if YAVG:
                                coupled_summary["uY"] = np.asarray(uY, np.float32)
                    # --- theta negotiation diagnostics (per-iter) ---
                    if bool(getattr(problem.flags, "enable_theta", False)):
                        dbg = getattr(problem, "_theta_dbg", {}) or {}
                        NN = int(problem.N)

                        def pack(key, fill=np.nan, dtype=np.float32):
                            v = np.full((NN,), fill, dtype=dtype)
                            for ii in range(NN):
                                di = dbg.get(ii, None)
                                if not di:
                                    continue
                                if key in di:
                                    v[ii] = di[key]
                            return v

                        theta_summary = {
                            # 1) 协商程度
                            "theta_H": pack("H"),
                            "theta_max": pack("max_th"),
                            "theta_argmax": pack("argmax", fill=-1, dtype=np.int32),
                            "theta_delta_l2": pack("delta_l2"),
                            # 2) 一致性牵引残差
                            "theta_res_y": pack("res_y"),
                            # 3) 量级对比
                            "theta_pull": pack("pull"),
                            "theta_c_std": pack("c_std"),
                            "theta_c_rng": pack("c_rng"),
                            "theta_ratio": pack("ratio"),
                        }

                        coupled_summary.update(theta_summary)


                    # YAVG diagnostics (small scalars)
                    if YAVG:
                        coupled_summary.update({
                            "YAVG_rY": float(rY),
                            "YAVG_sY": float(sY),
                            "YAVG_uY_norm": float(np.linalg.norm(uY, ord=2)),
                        })

                    run_log.append_iter(
                        k=int(k),
                        eta=float(eta),
                        r_stop=float(r_stop),
                        s_stop=float(s_stop),
                        eps_pri=float(eps_pri_val),
                        eps_dual=float(eps_dual_val),
                        block_order=list(run_log.block_order),  # if getattr(run_log, "block_order", None) else list(reg.names()),
                        r_by_block={kk: float(vv) for kk, vv in dict(r_blk).items()},
                        s_by_block={kk: float(vv) for kk, vv in dict(s_blk).items()},
                        proj_violation={kk: float(vv) for kk, vv in dict(proj_violation).items()},
                        active_mask=active_mask,
                        u_mask=u_mask,
                        coupled_summary=coupled_summary,
                        z=z,
                        q=q,
                        u=u,
                        store_blocks=store_blocks,
                        store_full_state=store_full_state,
                        state_dtype=state_dtype,
                    )
            except Exception:
                pass

        if (r_stop <= eps_pri_val) and (s_stop <= eps_dual_val):
            break


        # --- residual balancing for NEXT iteration (optional) ---
        if enable_rb:
            # eta_new, u_scaled, info = residual_balance_step(eta, u, r_global, s_global, problem.params, problem.flags)
            # eta_new, u_scaled, info = residual_balance_step(eta, u, r_stop, s_stop, problem.params, problem.flags)
            # (6.8) In YAVG, do residual balancing ONLY on eq part (avoid Y scaling confusing balancing).
            if YAVG and theory_mode and (eqm is not None):
                eta_new, u_scaled, info = residual_balance_step(eta, u, r_eq, s_eq, problem.params, problem.flags)
            else:
                eta_new, u_scaled, info = residual_balance_step(eta, u, r_stop, s_stop, problem.params, problem.flags)

            if info.get("changed", False):
                rb_events.append(info)
            eta = float(eta_new)
            u = np.asarray(u_scaled, dtype=np.float32)

            if YAVG:
                u[:, sl_y] = 0.0

    # next_pos = decode_pos(reg, z, problem.N, problem.coord_dim) if "pos" in reg.names() else np.asarray(problem.robot_pos, dtype=np.float32)

    # Theory mode freezes z for non-macro blocks; next_pos must come from each robot's local q_i.
    if ("pos" in reg.names()) and theory_mode:
        sl = reg.sl("pos")
        pos_dim = int(problem.coord_dim)
        pos_stack = np.zeros((problem.N, pos_dim), dtype=np.float32)
        # Each agent i decides its own slot (i) in the stacked pos block.
        for i in range(problem.N):
            blk = np.asarray(q[i, sl], dtype=np.float32).reshape((problem.N, pos_dim))
            pos_stack[i] = blk[i]
        next_pos = pos_stack
    else:
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

    # Attach global Y dual to outputs without breaking dataclass compatibility.
    if YAVG:
        setattr(sol, "uY", np.asarray(uY, dtype=np.float32).copy())
        setattr(ws, "uY", np.asarray(uY, dtype=np.float32).copy())
        setattr(diag, "uY", np.asarray(uY, dtype=np.float32).copy())
    # attach optional extra diagnostics (doesn't change dataclass definition)
    if enable_async:
        setattr(diag, "async_active_hist", async_active_hist)
    if enable_rb:
        setattr(diag, "residual_balance_events", rb_events)

    # window info is better recorded at entry level (per env step), but we keep hook
    
    # --- optional logging finalize ---
    if run_log is not None:
        try:
            run_log.finalize(sol=sol, diag=diag)
        except Exception:
            pass

    return sol, ws, diag

