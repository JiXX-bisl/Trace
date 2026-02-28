from __future__ import annotations
import sys

"""
Scene S2 validation runner (v2):
- Modes:
    * stability  : sanity + feasibility + residual trend (no strict eps)
    * strict     : tries to enforce ADMM-like convergence (disable q-step hard updates by default)
    * functional : validate objective improvement vs stay-put and compare to oracle (no strict eps)
- Always runs S2 builder sanity checks (LoS/NLoS, hops/capacity, candidate-domain free-space).
- Writes:
    outputs/scene_s2/summary.json
    outputs/scene_s2/results.npz
    (optional) curve pngs
"""

import argparse
import json
import os
import time
from dataclasses import asdict, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from scripts.core.data import CadmmParams, FeatureFlags, CadmmWarmStart
from scripts.core.cadmm_solver import build_problem_snapshot, solve_inner_cadmm
from scripts.core.inner.blocks import make_registry

from scripts.experiments.inner_scene_builder import make_scene_s2_problem


# ----------------------------
# Helpers: geometry & indexing
# ----------------------------
def world_to_vox_xyz_floor(xyz: np.ndarray, *, resolution: float, nx: int, ny: int, nz: int) -> Tuple[int, int, int]:
    x = float(xyz[0]); y = float(xyz[1]); z = float(xyz[2])
    ix = int(np.clip(np.floor(x / resolution), 0, nx - 1))
    iy = int(np.clip(np.floor(y / resolution), 0, ny - 1))
    iz = int(np.clip(np.floor(z / resolution), 0, nz - 1))
    return ix, iy, iz


_OFFSETS_CACHE_3D = {}
_OFFSETS_CACHE_2D = {}

def _ball_offsets(rv: int) -> np.ndarray:
    rv = int(rv)
    if rv in _OFFSETS_CACHE_3D:
        return _OFFSETS_CACHE_3D[rv]
    pts = []
    r2 = rv * rv
    for dz in range(-rv, rv + 1):
        for dy in range(-rv, rv + 1):
            for dx in range(-rv, rv + 1):
                if dx * dx + dy * dy + dz * dz <= r2:
                    pts.append((dz, dy, dx))
    arr = np.asarray(pts, dtype=np.int32)
    _OFFSETS_CACHE_3D[rv] = arr
    return arr

def _disk_offsets(rv: int) -> np.ndarray:
    rv = int(rv)
    if rv in _OFFSETS_CACHE_2D:
        return _OFFSETS_CACHE_2D[rv]
    pts = []
    r2 = rv * rv
    for dy in range(-rv, rv + 1):
        for dx in range(-rv, rv + 1):
            if dx * dx + dy * dy <= r2:
                pts.append((dy, dx))
    arr = np.asarray(pts, dtype=np.int32)
    _OFFSETS_CACHE_2D[rv] = arr
    return arr

def entropy_gain_in_sensing_region(
    ent: np.ndarray,
    cand_xyz: np.ndarray,
    *,
    sense_radius: float = 5.0,
    resolution: float = 1.0,
) -> float:
    ent = np.asarray(ent, dtype=np.float32)
    x = float(cand_xyz[0])
    y = float(cand_xyz[1])
    z = float(cand_xyz[2] if cand_xyz.size >= 3 else 0.0)

    rv = int(np.ceil(float(sense_radius) / float(resolution)))

    if ent.ndim == 3:
        D, H, W = int(ent.shape[0]), int(ent.shape[1]), int(ent.shape[2])
        cx = int(np.clip(np.floor(x / resolution), 0, W - 1))
        cy = int(np.clip(np.floor(y / resolution), 0, H - 1))
        cz = int(np.clip(np.floor(z / resolution), 0, D - 1))
        offs = _ball_offsets(rv)
        s = 0.0
        n = 0
        for dz, dy, dx in offs:
            zz = cz + int(dz)
            yy = cy + int(dy)
            xx = cx + int(dx)
            if 0 <= zz < D and 0 <= yy < H and 0 <= xx < W:
                v = float(ent[zz, yy, xx])
                if np.isfinite(v):
                    s += v
                    n += 1
        return (s / float(n)) if n > 0 else 0.0

    if ent.ndim == 2:
        H, W = int(ent.shape[0]), int(ent.shape[1])
        cx = int(np.clip(np.floor(x / resolution), 0, W - 1))
        cy = int(np.clip(np.floor(y / resolution), 0, H - 1))
        offs = _disk_offsets(rv)
        s = 0.0
        n = 0
        for dy, dx in offs:
            yy = cy + int(dy)
            xx = cx + int(dx)
            if 0 <= yy < H and 0 <= xx < W:
                v = float(ent[yy, xx])
                if np.isfinite(v):
                    s += v
                    n += 1
        return (s / float(n)) if n > 0 else 0.0

    return 0.0

def gain_at(ent_zyx: np.ndarray, p_xyz: np.ndarray, *, resolution: float = 1.0) -> float:
    return float(entropy_gain_in_sensing_region(ent_zyx, p_xyz, sense_radius=5.0, resolution=resolution))


def seg_intersect_aabb(p0: np.ndarray, p1: np.ndarray, bmin: np.ndarray, bmax: np.ndarray, eps: float = 1e-9) -> bool:
    p0 = np.asarray(p0, dtype=np.float32).reshape(3)
    p1 = np.asarray(p1, dtype=np.float32).reshape(3)
    d = p1 - p0
    tmin, tmax = 0.0, 1.0
    for k in range(3):
        dk = float(d[k])
        if abs(dk) < eps:
            if (float(p0[k]) < float(bmin[k])) or (float(p0[k]) > float(bmax[k])):
                return False
        else:
            inv = 1.0 / dk
            t1 = (float(bmin[k]) - float(p0[k])) * inv
            t2 = (float(bmax[k]) - float(p0[k])) * inv
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return False
    return True


def los_clear(p0: np.ndarray, p1: np.ndarray, obstacles: List[Any]) -> bool:
    for box in obstacles:
        bmin = np.array([float(box.x_min), float(box.y_min), float(box.z_min)], dtype=np.float32)
        bmax = np.array([float(box.x_max), float(box.y_max), float(box.z_max)], dtype=np.float32)
        if seg_intersect_aabb(p0, p1, bmin, bmax):
            return False
    return True


def edges_by_los_threshold(robot_pos: np.ndarray, obstacles: List[Any], *, comm_threshold: float) -> np.ndarray:
    rp = np.asarray(robot_pos, dtype=np.float32).reshape(-1, 3)
    N = int(rp.shape[0])
    edges: List[Tuple[int, int]] = []
    for i in range(N):
        for j in range(i + 1, N):
            d = float(np.linalg.norm(rp[i] - rp[j]))
            if d > float(comm_threshold):
                continue
            if not los_clear(rp[i], rp[j], obstacles):
                continue
            edges.append((i, j))
    return np.asarray(edges, dtype=np.int32) if edges else np.zeros((0, 2), dtype=np.int32)


def hops_matrix_from_edges(N: int, edges: np.ndarray) -> np.ndarray:
    N = int(N)
    adj = [[] for _ in range(N)]
    for (i, j) in np.asarray(edges, dtype=np.int32).reshape(-1, 2):
        adj[int(i)].append(int(j))
        adj[int(j)].append(int(i))
    hops = -np.ones((N, N), dtype=np.int32)
    for s in range(N):
        hops[s, s] = 0
        q = [s]
        head = 0
        while head < len(q):
            u = q[head]
            head += 1
            for v in adj[u]:
                if hops[s, v] < 0:
                    hops[s, v] = hops[s, u] + 1
                    q.append(v)
    return hops


def pair_capacity_from_pos_and_hops(robot_pos: np.ndarray, hops: np.ndarray, *, Cmax: float) -> np.ndarray:
    rp = np.asarray(robot_pos, dtype=np.float32).reshape(-1, 3)
    N = int(rp.shape[0])
    out = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            h = int(hops[i, j])
            if h <= 0:
                continue
            d = float(np.linalg.norm(rp[i] - rp[j]))
            if d <= 1e-6:
                continue
            out[i, j] = float(Cmax) / (d * float(h))
    return out


def snap_pos_to_candidates(z_pos: np.ndarray, candidate_moves: List[np.ndarray]) -> np.ndarray:
    rp = np.asarray(z_pos, dtype=np.float32).reshape(-1, 3)
    out = np.zeros_like(rp)
    for i in range(rp.shape[0]):
        cands = np.asarray(candidate_moves[i], dtype=np.float32).reshape(-1, 3)
        if cands.size == 0:
            out[i] = rp[i]
            continue
        d2 = np.sum((cands - rp[i][None, :]) ** 2, axis=1)
        out[i] = cands[int(np.argmin(d2))]
    return out.astype(np.float32, copy=False)


def objective_score(
    pos: np.ndarray,
    ent_zyx: np.ndarray,
    obstacles: List[Any],
    *,
    resolution: float,
    comm_threshold: float,
    Cmax: float,
    root_id: int,
    w_ent: float,
    w_cap: float,
    w_conn_penalty: float,
) -> Dict[str, float]:
    pos = np.asarray(pos, dtype=np.float32).reshape(-1, 3)
    N = int(pos.shape[0])

    edges = edges_by_los_threshold(pos, obstacles, comm_threshold=comm_threshold)
    hops = hops_matrix_from_edges(N, edges)
    pairC = pair_capacity_from_pos_and_hops(pos, hops, Cmax=Cmax)

    ent_sum = 0.0
    for i in range(N):
        ent_sum += gain_at(ent_zyx, pos[i], resolution=resolution)

    unreachable = 0
    cap_sum = 0.0
    for i in range(N):
        if i == root_id:
            continue
        if int(hops[i, root_id]) < 0:
            unreachable += 1
        else:
            cap_sum += float(np.log1p(float(pairC[i, root_id])))

    score = float(w_ent) * float(ent_sum) + float(w_cap) * float(cap_sum) - float(w_conn_penalty) * float(unreachable)
    return {
        "score": float(score),
        "ent_sum": float(ent_sum),
        "cap_sum_log1p_to_root": float(cap_sum),
        "unreachable_to_root": float(unreachable),
        "num_edges_los": float(edges.shape[0]),
    }


# ----------------------------
# Instrumentation trace
# ----------------------------
class InnerTrace:
    def __init__(self) -> None:
        self.r_total: List[float] = []
        self.s_total: List[float] = []
        self.r_by_name: List[Dict[str, float]] = []
        self.s_by_name: List[Dict[str, float]] = []

        self.proj_pre_by_block: List[Dict[str, float]] = []
        self.proj_post_by_block: List[Dict[str, float]] = []
        self.proj_corr_by_block: List[Dict[str, float]] = []

        self.coupled_pass_sigma: List[bool] = []
        self.coupled_pass_f_hat: List[bool] = []
        self.budget_violation: List[float] = []


def run_inner_solver_with_trace(
    problem_snapshot: Any,
    *,
    warm_start: Optional[CadmmWarmStart],
    rng: np.random.Generator,
) -> Tuple[Any, CadmmWarmStart, Any, InnerTrace]:
    import scripts.core.cadmm_solver as cadmm_solver
    import scripts.core.inner.coupled_projection_passes as cpp
    import scripts.core.inner.qos_metrics as qos

    trace = InnerTrace()

    orig_make_residual_model = cadmm_solver.make_residual_model
    orig_apply_coupled = cpp.apply_coupled_projection_passes
    orig_update_budget_cache = qos.update_budget_cache

    class _LoggingResidual:
        def __init__(self, base):
            self._base = base

        def primal_block_norms(self, q, z, reg):
            norms, total = self._base.primal_block_norms(q, z, reg)
            trace.r_total.append(float(total))
            trace.r_by_name.append({k: float(v) for k, v in dict(norms).items()})
            return norms, total

        def dual_block_norms(self, z, z_prev, eta, reg):
            norms, total = self._base.dual_block_norms(z, z_prev, eta, reg)
            trace.s_total.append(float(total))
            trace.s_by_name.append({k: float(v) for k, v in dict(norms).items()})
            return norms, total

    def _make_residual_model_logged(problem, reg, flags):
        base = orig_make_residual_model(problem, reg, flags)
        return _LoggingResidual(base)

    def _apply_coupled_logged(problem, reg, z_blocks, diag_by_block, *, method, iters, tol):
        orig_apply_coupled(problem, reg, z_blocks, diag_by_block, method=method, iters=iters, tol=tol)

        pre: Dict[str, float] = {}
        post: Dict[str, float] = {}
        corr: Dict[str, float] = {}

        for bk, bv in dict(diag_by_block).items():
            bvd = dict(bv) if bv is not None else {}
            pre[bk] = float(bvd.get("pre_max_violation", 0.0) or 0.0)
            post[bk] = float(bvd.get("max_violation", 0.0) or 0.0)
            corr[bk] = float(bvd.get("corr_norm", 0.0) or 0.0)

        trace.proj_pre_by_block.append(pre)
        trace.proj_post_by_block.append(post)
        trace.proj_corr_by_block.append(corr)

        trace.coupled_pass_sigma.append(bool("coupled_pass" in dict(diag_by_block.get("sigma", {}))))
        trace.coupled_pass_f_hat.append(bool("coupled_pass" in dict(diag_by_block.get("f_hat", {}))))

    def _update_budget_cache_logged(*, window_state, reg, value, ema, enable_update, enable_soft_violation, tol, flags=None):
        orig_update_budget_cache(
            window_state=window_state,
            reg=reg,
            value=value,
            ema=ema,
            enable_update=enable_update,
            enable_soft_violation=enable_soft_violation,
            tol=tol,
            flags=flags,
        )
        try:
            viol = float(getattr(window_state, "budget_violation", 0.0)) if window_state is not None else 0.0
        except Exception:
            viol = 0.0
        trace.budget_violation.append(float(viol))

    cadmm_solver.make_residual_model = _make_residual_model_logged
    cpp.apply_coupled_projection_passes = _apply_coupled_logged
    qos.update_budget_cache = _update_budget_cache_logged

    try:
        sol, ws, diag = solve_inner_cadmm(problem_snapshot, warm_start, rng)
    finally:
        cadmm_solver.make_residual_model = orig_make_residual_model
        cpp.apply_coupled_projection_passes = orig_apply_coupled
        qos.update_budget_cache = orig_update_budget_cache

    K = int(getattr(diag, "iters", len(trace.r_total)))
    trace.r_total = trace.r_total[:K]
    trace.s_total = trace.s_total[:K]
    trace.r_by_name = trace.r_by_name[:K]
    trace.s_by_name = trace.s_by_name[:K]
    trace.proj_pre_by_block = trace.proj_pre_by_block[:K]
    trace.proj_post_by_block = trace.proj_post_by_block[:K]
    trace.proj_corr_by_block = trace.proj_corr_by_block[:K]
    trace.coupled_pass_sigma = trace.coupled_pass_sigma[:K]
    trace.coupled_pass_f_hat = trace.coupled_pass_f_hat[:K]
    trace.budget_violation = trace.budget_violation[:K]

    return sol, ws, diag, trace


# ----------------------------
# Defaults
# ----------------------------
def default_flags_s2() -> FeatureFlags:
    f = FeatureFlags()
    f = replace(
        f,
        coord_dim=3,
        use_linear_residual=True,
        linear_assembly_mode="coupled_basic",
        enable_coupled_groups=True,
        include_coupled_groups_in_stop=False,
        enable_sigma_coupled_to_y=True,
        enable_flow_coupled_to_budget=True,
        enable_budget_cache_update=True,
        enable_budget_soft_violation=True,
        enable_residual_balancing=True,
        enable_over_relax=False,
        proj_tol=1e-6,
        enable_qstep_damped_y_hat=True,
        enable_qstep_damped_sigma=True
    )
    return f


def disable_qstep_hard_updates(flags: FeatureFlags) -> FeatureFlags:
    # This makes q-step closer to "q=z-u" style, enabling strict convergence checks.
    return replace(
        flags,
        enable_qstep_update_y_hat=False,
        enable_qstep_update_sigma=False,
        enable_qstep_update_B_hat=False,
        enable_qstep_update_f_hat=False,
        enable_qstep_update_r_hat=False,
    )


def default_params_s2(*, k_max: int, eps: float, activate_objectives: bool) -> CadmmParams:
    p = CadmmParams(
        eta=1.0,
        alpha=1.0,
        eps_pri=float(eps),
        eps_dual=float(eps),
        k_max=int(k_max),
        ttl_hops=2,
        t_fresh=2,
        mu=2.0,
        tau_incr=2.0,
        tau_decr=2.0,
        move_w=0.0,
        cover_gamma=0.0,
        new_w=1.0 if activate_objectives else 0.0,   # <- entropy term
        fe_w=0.0,
        role_task_weight=1.0,
        role_relay_weight=1.0,
        role_main_weight=1.0,
        qos_exec_weight=1.0,
        qos_relay_weight=1.0,
        qos_default_weight=1.0,
        qos_break_w=0.0,
        qos_improve_w=0.0,
        qos_degrade_w=0.0,
        urgency_w=0.5 if activate_objectives else 0.0,  # <- task urgency
        rep_w=0.05 if activate_objectives else 0.0,     # <- repulsion
        rep_sigma=1.0,
        budget_cache_ema=0.2,
    )
    setattr(p, "eta_stability_ratio", 100.0)
    setattr(p, "root_id_default", 0)
    return p


# ----------------------------
# S2 sanity checks
# ----------------------------
def check_s2_builder_and_snapshot(
    env: Any,
    snapshot: Any,
    *,
    comm_threshold: float,
    Cmax: float,
    resolution: float,
    step_len_ugv: float,
    step_len_uav: float,
    ugv_ids: Tuple[int, int, int] = (0, 1, 2),
) -> Tuple[List[str], Dict[str, Any]]:
    reasons: List[str] = []
    stats: Dict[str, Any] = {}

    window = getattr(env, "window", None)
    occ = getattr(window, "scene_occupancy", None)
    obstacles = getattr(window, "scene_obstacles", None)
    hops_mat = getattr(window, "hops_mat", None)
    pair_cap = getattr(window, "pair_capacity", None)

    ent = np.asarray(getattr(snapshot, "frontier_entropy", np.zeros((0,), dtype=np.float32)))
    rp = np.asarray(getattr(snapshot, "robot_pos", np.zeros((0, 3), dtype=np.float32))).reshape(-1, 3)

    stats["N"] = int(getattr(snapshot, "N", rp.shape[0]))
    stats["E"] = int(getattr(snapshot, "E", 0))
    stats["coverage"] = float(getattr(snapshot, "coverage", 0.0))

    if ent.ndim != 3:
        reasons.append("frontier_entropy_not_3d")
    else:
        D, H, W = int(ent.shape[0]), int(ent.shape[1]), int(ent.shape[2])
        stats["entropy_shape"] = [D, H, W]
        stats["entropy_mean"] = float(np.mean(ent))
        stats["entropy_min"] = float(np.min(ent))
        stats["entropy_max"] = float(np.max(ent))
        if stats["entropy_min"] < -1e-6 or stats["entropy_max"] > 1.0 + 1e-6:
            reasons.append("entropy_values_out_of_[0,1]_range")

    if occ is None:
        reasons.append("window_missing_scene_occupancy")
    else:
        occ = np.asarray(occ)
        stats["occupancy_shape_xyz"] = list(occ.shape)
        stats["free_ratio"] = float(np.mean(occ == 0))

    if obstacles is None:
        reasons.append("window_missing_scene_obstacles")
        obstacles = []
    else:
        stats["num_obstacles"] = int(len(obstacles))

    edges = np.asarray(getattr(snapshot.link, "edges", np.zeros((0, 2), dtype=np.int32)), dtype=np.int32).reshape(-1, 2)
    bad_edges = 0
    for (i, j) in edges:
        i = int(i); j = int(j)
        d = float(np.linalg.norm(rp[i] - rp[j]))
        if d > comm_threshold + 1e-6:
            bad_edges += 1
            continue
        if not los_clear(rp[i], rp[j], obstacles):
            bad_edges += 1
            continue
    stats["bad_los_edges"] = int(bad_edges)
    if bad_edges > 0:
        reasons.append("los_edges_invalid_dist_or_blocked")

    if hops_mat is None or pair_cap is None:
        reasons.append("window_missing_hops_or_pair_capacity")
    else:
        hops_mat = np.asarray(hops_mat, dtype=np.int32)
        pair_cap = np.asarray(pair_cap, dtype=np.float32)

        hops_ref = hops_matrix_from_edges(int(rp.shape[0]), edges)
        if hops_ref.shape != hops_mat.shape or np.any(hops_ref != hops_mat):
            reasons.append("hops_mat_mismatch_bfs")
        else:
            rng = np.random.default_rng(0)
            mism = 0
            N = int(rp.shape[0])
            for _ in range(30):
                a = int(rng.integers(0, N))
                b = int(rng.integers(0, N))
                if a == b:
                    continue
                h = int(hops_mat[a, b])
                d = float(np.linalg.norm(rp[a] - rp[b]))
                expected = 0.0
                if h > 0 and d > 1e-6:
                    expected = float(Cmax) / (d * float(h))
                got = float(pair_cap[a, b])
                if abs(got - expected) > 1e-3 * max(1.0, expected):
                    mism += 1
            stats["pair_capacity_mismatch_samples"] = int(mism)
            if mism > 0:
                reasons.append("pair_capacity_formula_mismatch")

    cand_list = getattr(snapshot, "candidate_moves", None)
    if cand_list is None:
        reasons.append("candidate_moves_missing")
    else:
        cand_sizes = []
        bad = 0
        for rid, cands in enumerate(cand_list):
            c = np.asarray(cands, dtype=np.float32).reshape(-1, 3)
            cand_sizes.append(int(c.shape[0]))
            if c.shape[0] == 0:
                bad += 1
                continue
            delta = c - rp[rid][None, :]
            if rid in ugv_ids:
                if np.any(np.abs(c[:, 2]) > 1e-3):
                    bad += 1
                r_xy = np.sqrt(delta[:, 0] ** 2 + delta[:, 1] ** 2)
                if float(np.max(r_xy)) > float(step_len_ugv) + 1e-3:
                    bad += 1
            else:
                r3 = np.linalg.norm(delta, axis=1)
                if float(np.max(r3)) > float(step_len_uav) + 1e-3:
                    bad += 1

            if occ is not None:
                nx, ny, nz = occ.shape
                for p in c[: min(20, c.shape[0])]:
                    ix, iy, iz = world_to_vox_xyz_floor(p, resolution=resolution, nx=nx, ny=ny, nz=nz)
                    if int(occ[ix, iy, iz]) != 0:
                        bad += 1
                        break

        stats["candidate_sizes"] = cand_sizes
        stats["bad_candidate_sets"] = int(bad)
        if bad > 0:
            reasons.append("candidate_moves_invalid_or_not_free")

    return reasons, stats


# ----------------------------
# Evaluation modes
# ----------------------------
def eval_stability(trace: InnerTrace, diag: Any) -> Tuple[List[str], Dict[str, Any]]:
    reasons: List[str] = []
    stats: Dict[str, Any] = {}

    r = np.asarray(trace.r_total, dtype=np.float32)
    s = np.asarray(trace.s_total, dtype=np.float32)
    K = int(getattr(diag, "iters", int(r.size)))
    stats["iters"] = int(K)

    if r.size >= 10:
        r_first = float(np.mean(r[:5])); r_last = float(np.mean(r[-5:]))
        stats["r_mean_first5"] = r_first
        stats["r_mean_last5"] = r_last
        if not (r_last < r_first):
            reasons.append("r_total_not_decreasing")
    else:
        stats["r_mean_first5"] = float(np.mean(r)) if r.size else 0.0
        stats["r_mean_last5"] = float(np.mean(r)) if r.size else 0.0

    if s.size >= 10:
        s_first = float(np.mean(s[:5])); s_last = float(np.mean(s[-5:]))
        stats["s_mean_first5"] = s_first
        stats["s_mean_last5"] = s_last
        if not (s_last < s_first):
            reasons.append("s_total_not_decreasing")
    else:
        stats["s_mean_first5"] = float(np.mean(s)) if s.size else 0.0
        stats["s_mean_last5"] = float(np.mean(s)) if s.size else 0.0

    # feasibility post-violation (very important for stability)
    if trace.proj_post_by_block:
        last = trace.proj_post_by_block[-1]
        for k in ("sigma", "f_hat", "B_hat", "y_hat"):
            stats[f"proj_post_{k}_last"] = float(last.get(k, 0.0))
        # loose threshold
        if any(float(last.get(k, 0.0)) > 1e-4 for k in ("sigma", "f_hat", "B_hat", "y_hat")):
            reasons.append("proj_post_violation_not_small")

    return reasons, stats


def eval_strict(trace: InnerTrace, diag: Any, *, eps: float, abs_factor: float, allow_hit_kmax: bool) -> Tuple[List[str], Dict[str, Any]]:
    reasons, stats = eval_stability(trace, diag)

    r = np.asarray(trace.r_total, dtype=np.float32)
    s = np.asarray(trace.s_total, dtype=np.float32)
    K = int(getattr(diag, "iters", int(r.size)))
    k_max = int(getattr(getattr(diag, "_params", None), "k_max", 0) or 0)

    # detect hit_kmax by comparing to requested K bound from trace length
    # (caller will supply k_max in stats anyway; keep simple)
    stats["hit_kmax"] = False
    if k_max > 0 and K >= k_max:
        stats["hit_kmax"] = True
        if not allow_hit_kmax:
            reasons.append("hit_kmax_no_convergence")

    thr = float(abs_factor) * float(eps)
    stats["thr_r"] = float(thr)
    stats["thr_s"] = float(thr)

    if r.size and float(np.mean(r[-5:])) > thr:
        reasons.append("r_not_small_enough")
    if s.size and float(np.mean(s[-5:])) > thr:
        reasons.append("s_not_small_enough")

    return reasons, stats


def eval_functional(
    snapshot: Any,
    sol: Any,
    env: Any,
    *,
    resolution: float,
    comm_threshold: float,
    Cmax: float,
    obj_w_ent: float,
    obj_w_cap: float,
    obj_w_conn_penalty: float,
) -> Tuple[List[str], Dict[str, Any]]:
    reasons: List[str] = []
    stats: Dict[str, Any] = {}

    ent = np.asarray(snapshot.frontier_entropy, dtype=np.float32)
    obstacles = getattr(env.window, "scene_obstacles", [])
    root_id = int(getattr(snapshot.params, "root_id_default", 0))

    reg = make_registry(snapshot.N, snapshot.E, snapshot.G, snapshot.flags)
    z_blocks = reg.unpack(sol.z)
    z_pos = np.asarray(z_blocks.get("pos", snapshot.robot_pos.reshape(-1)), dtype=np.float32).reshape(snapshot.N, 3)
    z_snap = snap_pos_to_candidates(z_pos, snapshot.candidate_moves)

    score_snap = objective_score(
        z_snap, ent, obstacles,
        resolution=resolution,
        comm_threshold=comm_threshold,
        Cmax=Cmax,
        root_id=root_id,
        w_ent=obj_w_ent,
        w_cap=obj_w_cap,
        w_conn_penalty=obj_w_conn_penalty,
    )
    score_stay = objective_score(
        np.asarray(snapshot.robot_pos, dtype=np.float32).reshape(snapshot.N, 3),
        ent, obstacles,
        resolution=resolution,
        comm_threshold=comm_threshold,
        Cmax=Cmax,
        root_id=root_id,
        w_ent=obj_w_ent,
        w_cap=obj_w_cap,
        w_conn_penalty=obj_w_conn_penalty,
    )

    stats["obj_admm_snap"] = score_snap
    stats["obj_stay_put"] = score_stay

    oracle_score = getattr(env.window, "oracle_score", None)
    if oracle_score is not None:
        stats["obj_oracle_score"] = float(oracle_score)
        if abs(float(oracle_score)) > 1e-9:
            stats["obj_ratio_to_oracle"] = float(score_snap["score"] / float(oracle_score))

    # Require improvement over stay-put (soft but meaningful)
    if score_snap["score"] + 1e-9 < score_stay["score"]:
        reasons.append("objective_worse_than_stay_put")

    return reasons, stats

def _pack_norm_series(dict_list: List[Dict[str, float]], K: int) -> Tuple[np.ndarray, np.ndarray]:
    """dict_list[i] = {block_name: norm}. Return (keys, mat[K,B])."""
    keys_set = set()
    for d in dict_list[:K]:
        keys_set.update(d.keys())
    keys = sorted(keys_set)

    mat = np.zeros((K, len(keys)), dtype=np.float32)
    for i in range(K):
        d = dict_list[i] if i < len(dict_list) else {}
        for j, k in enumerate(keys):
            mat[i, j] = float(d.get(k, 0.0))
    return np.asarray(keys, dtype=str), mat
# ----------------------------
# Plotting
# ----------------------------
def plot_curves(out_dir: str, trace: InnerTrace, diag: Any) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    os.makedirs(out_dir, exist_ok=True)
    K = int(getattr(diag, "iters", len(trace.r_total)))
    xs = np.arange(K)

    # ----------------
    # 1) r / s
    # ----------------
    r = np.asarray(trace.r_total[:K], dtype=np.float32)
    s = np.asarray(trace.s_total[:K], dtype=np.float32)

    plt.figure()
    plt.plot(xs, r, label="r_total")
    plt.plot(xs, s, label="s_total")
    plt.xlabel("inner_iter")
    plt.ylabel("norm")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "curve_r_s.png"), dpi=180)
    plt.close()

    # ----------------
    # 2) eta (from diag.eta_hist; fallback to fixed params.eta)
    # ----------------
    eta = np.asarray(getattr(diag, "eta_hist", [])[:K], dtype=np.float32)
    if eta.size == 0:
        # strict runs often disable residual balancing => eta_hist may be empty
        p = getattr(diag, "_params", None)
        eta0 = float(getattr(p, "eta", 1.0)) if p is not None else 1.0
        eta = np.full((K,), eta0, dtype=np.float32)

    plt.figure()
    plt.plot(xs[: eta.size], eta, label="eta")
    plt.xlabel("inner_iter")
    plt.ylabel("eta")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "curve_eta.png"), dpi=180)
    plt.close()

    # ----------------
    # 3) projection post violation (你原来就有)
    # ----------------
    if trace.proj_post_by_block:
        keys = ["sigma", "f_hat", "B_hat", "y_hat"]
        plt.figure()
        for k in keys:
            ys = np.array([float(d.get(k, 0.0)) for d in trace.proj_post_by_block[:K]], dtype=np.float32)
            # strict 时可能全为 0，这里也允许画（至少 y_hat）
            if k == "y_hat" or np.any(ys != 0.0):
                plt.plot(xs, ys, label=f"post_{k}")
        plt.xlabel("inner_iter")
        plt.ylabel("post max_violation")
        plt.yscale("symlog", linthresh=1e-8)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "curve_proj_post.png"), dpi=180)
        plt.close()

    # ----------------
    # 4) projection correction magnitude (corr_norm)  <-- 新增：corr_y_hat 就在这里
    # ----------------
    if trace.proj_corr_by_block:
        keys = ["sigma", "f_hat", "B_hat", "y_hat"]
        plt.figure()
        for k in keys:
            ys = np.array([float(d.get(k, 0.0)) for d in trace.proj_corr_by_block[:K]], dtype=np.float32)
            # corr_y_hat 我们强制画出来；其它块非零再画
            if k == "y_hat" or np.any(ys != 0.0):
                plt.plot(xs, ys, label=f"corr_{k}")
        plt.xlabel("inner_iter")
        plt.ylabel("corr_norm")
        plt.yscale("symlog", linthresh=1e-8)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "curve_proj_corr.png"), dpi=180)
        plt.close()

    if trace.proj_pre_by_block:
        keys = ["sigma", "f_hat", "B_hat", "y_hat"]
        plt.figure()
        for k in keys:
            ys = np.array([float(d.get(k, 0.0)) for d in trace.proj_pre_by_block[:K]], dtype=np.float32)
            if k == "y_hat" or np.any(ys != 0.0):
                plt.plot(xs, ys, label=f"pre_{k}")
        plt.xlabel("inner_iter")
        plt.ylabel("pre max_violation")
        plt.yscale("symlog", linthresh=1e-8)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "curve_proj_pre.png"), dpi=180)
        plt.close()


# ----------------------------
# Visualization bundle export
# ----------------------------
def _variant_tag_from_args(args: Any) -> str:
    parts: List[str] = []
    if bool(getattr(args, "disable_residual_balancing", False)):
        parts.append("noRB")
    if bool(getattr(args, "disable_flow_budget_coupling", False)):
        parts.append("noFB")
    if bool(getattr(args, "disable_sigma_y_coupling", False)):
        parts.append("noSY")
    return "_".join(parts) if parts else "full"


def _pack_candidates_padded(candidate_moves: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack ragged candidate list into (cand_pad[N,M,3], cand_mask[N,M], cand_sizes[N])."""
    N = int(len(candidate_moves))
    sizes = np.zeros((N,), dtype=np.int32)
    maxM = 0
    for i in range(N):
        c = np.asarray(candidate_moves[i], dtype=np.float32).reshape(-1, 3)
        sizes[i] = int(c.shape[0])
        maxM = max(maxM, int(c.shape[0]))
    cand_pad = np.full((N, int(maxM), 3), np.nan, dtype=np.float32)
    cand_mask = np.zeros((N, int(maxM)), dtype=np.uint8)
    for i in range(N):
        c = np.asarray(candidate_moves[i], dtype=np.float32).reshape(-1, 3)
        m = int(c.shape[0])
        if m > 0:
            cand_pad[i, :m, :] = c
            cand_mask[i, :m] = 1
    return cand_pad, cand_mask, sizes


def _closest_candidate_indices(pos: np.ndarray, cand_pad: np.ndarray, cand_mask: np.ndarray) -> np.ndarray:
    """Return argmin candidate index per robot for a snapped position."""
    pos = np.asarray(pos, dtype=np.float32).reshape(-1, 3)
    N = int(pos.shape[0])
    out = -np.ones((N,), dtype=np.int32)
    for i in range(N):
        mask = cand_mask[i].astype(bool)
        if not np.any(mask):
            continue
        c = cand_pad[i, mask, :]
        d2 = np.sum((c - pos[i][None, :]) ** 2, axis=1)
        j = int(np.argmin(d2))
        out[i] = int(np.nonzero(mask)[0][j])
    return out


def save_viz_bundle(
    out_dir: str,
    *,
    step: int,
    args: Any,
    env: Any,
    snapshot: Any,
    sol: Any,
    diag: Any,
    params: CadmmParams,
    flags: FeatureFlags,
    resolution: float,
    comm_threshold: float,
    Cmax: float,
) -> str:
    """
    Save a self-contained npz for thesis plotting:
      - 3D scene render: occupancy_xyz, obstacles_aabb, robot_pos, links
      - convergence curves: r/s, eta, projection pre/post/corr (already in results_step*.npz)
      - output action: z_pos (continuous), z_pos_snap, candidate indices
      - objective components: stay-put vs admm_snap (+ oracle if present)
    """
    os.makedirs(out_dir, exist_ok=True)
    window = getattr(env, "window", None)

    # Scene
    occ = np.asarray(getattr(window, "scene_occupancy", np.zeros((0, 0, 0), dtype=np.uint8)), dtype=np.uint8)
    obstacles = list(getattr(window, "scene_obstacles", []) or [])
    obs_aabb = np.zeros((len(obstacles), 6), dtype=np.float32)
    for i, box in enumerate(obstacles):
        obs_aabb[i, :] = np.array(
            [float(box.x_min), float(box.x_max), float(box.y_min), float(box.y_max), float(box.z_min), float(box.z_max)],
            dtype=np.float32,
        )

    scene_meta = getattr(window, "scene_meta", {}) or {}
    scene_meta_json = json.dumps(scene_meta, ensure_ascii=False)

    # Snapshot inputs
    robot_pos0 = np.asarray(getattr(snapshot, "robot_pos", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32).reshape(-1, 3)
    ent_zyx = np.asarray(getattr(snapshot, "frontier_entropy", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
    coverage = float(getattr(snapshot, "coverage", 0.0))
    rep_grad = np.asarray(getattr(snapshot, "repulsion_grad", np.zeros_like(robot_pos0)), dtype=np.float32).reshape(-1, 3)

    # Tasks (optional)
    task = getattr(snapshot, "task", None)
    task_pos = np.asarray(getattr(task, "task_pos", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32).reshape(-1, 3) if task is not None else np.zeros((0, 3), dtype=np.float32)
    task_deadline = np.asarray(getattr(task, "deadline", np.zeros((task_pos.shape[0],), dtype=np.float32)), dtype=np.float32).reshape(-1) if task is not None else np.zeros((0,), dtype=np.float32)
    task_priority = np.asarray(getattr(task, "priority", np.zeros((task_pos.shape[0],), dtype=np.float32)), dtype=np.float32).reshape(-1) if task is not None else np.zeros((0,), dtype=np.float32)
    task_cluster = np.asarray(getattr(task, "cluster_id", np.zeros((task_pos.shape[0],), dtype=np.int32)), dtype=np.int32).reshape(-1) if task is not None else np.zeros((0,), dtype=np.int32)

    # Candidate moves (packed)
    cand_pad, cand_mask, cand_sizes = _pack_candidates_padded(getattr(snapshot, "candidate_moves", []) or [])

    # Link snapshot (initial)
    link = getattr(snapshot, "link", None)
    edges0 = np.asarray(getattr(link, "edges", np.zeros((0, 2), dtype=np.int32)), dtype=np.int32).reshape(-1, 2) if link is not None else np.zeros((0, 2), dtype=np.int32)
    cap0 = np.asarray(getattr(link, "capacity", np.zeros((edges0.shape[0],), dtype=np.float32)), dtype=np.float32).reshape(-1) if link is not None else np.zeros((0,), dtype=np.float32)
    delay0 = np.asarray(getattr(link, "delay", np.zeros((edges0.shape[0],), dtype=np.float32)), dtype=np.float32).reshape(-1) if link is not None else np.zeros((0,), dtype=np.float32)
    plr0 = np.asarray(getattr(link, "plr", np.zeros((edges0.shape[0],), dtype=np.float32)), dtype=np.float32).reshape(-1) if link is not None else np.zeros((0,), dtype=np.float32)
    stale0 = np.asarray(getattr(link, "is_stale", np.zeros((edges0.shape[0],), dtype=np.int8)), dtype=np.int8).reshape(-1) if link is not None else np.zeros((0,), dtype=np.int8)

    hops0 = np.asarray(getattr(window, "hops_mat", np.zeros((robot_pos0.shape[0], robot_pos0.shape[0]), dtype=np.int32)), dtype=np.int32)
    pairC0 = np.asarray(getattr(window, "pair_capacity", np.zeros((robot_pos0.shape[0], robot_pos0.shape[0]), dtype=np.float32)), dtype=np.float32)

    # Oracle (optional)
    oracle_pos = getattr(window, "oracle_next_pos", None)
    oracle_score = getattr(window, "oracle_score", None)
    oracle_pos = np.asarray(oracle_pos, dtype=np.float32).reshape(-1, 3) if oracle_pos is not None else np.zeros((0, 3), dtype=np.float32)
    oracle_score = float(oracle_score) if oracle_score is not None else np.nan

    # Output action (continuous + snapped)
    reg = make_registry(snapshot.N, snapshot.E, snapshot.G, snapshot.flags)
    z_blocks = reg.unpack(sol.z)

    z_pos = np.asarray(z_blocks.get("pos", robot_pos0.reshape(-1)), dtype=np.float32).reshape(snapshot.N, 3)
    z_snap = snap_pos_to_candidates(z_pos, getattr(snapshot, "candidate_moves", []) or [])
    cand_idx = _closest_candidate_indices(z_snap, cand_pad, cand_mask)
    # Output link state at snapped action (for rendering / objective decomposition)
    edges1 = edges_by_los_threshold(z_snap, obstacles, comm_threshold=float(comm_threshold))
    hops1 = hops_matrix_from_edges(int(z_snap.shape[0]), edges1)
    pairC1 = pair_capacity_from_pos_and_hops(z_snap, hops1, Cmax=float(Cmax))

    # Objective components (same weights as functional defaults)
    root_id = int(getattr(snapshot.params, "root_id_default", 0))
    obj_snap = objective_score(
        z_snap, ent_zyx, obstacles,
        resolution=float(resolution),
        comm_threshold=float(comm_threshold),
        Cmax=float(Cmax),
        root_id=root_id,
        w_ent=float(getattr(args, "obj_w_ent", 1.0)),
        w_cap=float(getattr(args, "obj_w_cap", 0.2)),
        w_conn_penalty=float(getattr(args, "obj_w_conn_penalty", 50.0)),
    )
    obj_stay = objective_score(
        robot_pos0, ent_zyx, obstacles,
        resolution=float(resolution),
        comm_threshold=float(comm_threshold),
        Cmax=float(Cmax),
        root_id=root_id,
        w_ent=float(getattr(args, "obj_w_ent", 1.0)),
        w_cap=float(getattr(args, "obj_w_cap", 0.2)),
        w_conn_penalty=float(getattr(args, "obj_w_conn_penalty", 50.0)),
    )

    flags_json = json.dumps(to_jsonable_flags(flags), ensure_ascii=False)
    params_json = json.dumps(to_jsonable_params(params), ensure_ascii=False)

    # eta history is useful for RB ablation plots
    K = int(getattr(diag, "iters", 0) or 0)
    eta_hist = np.asarray(getattr(diag, "eta_hist", [])[:K], dtype=np.float32)

    variant = _variant_tag_from_args(args)
    out_path = os.path.join(out_dir, f"viz_step{int(step)}_{variant}.npz")

    save_full = bool(int(getattr(args, "save_full_solution", 0)))
    if save_full:
        sol_z = np.asarray(getattr(sol, "z", np.zeros((0,), dtype=np.float32)))
        sol_q = np.asarray(getattr(sol, "q", np.zeros((0,), dtype=np.float32)))
        sol_u = np.asarray(getattr(sol, "u", np.zeros((0,), dtype=np.float32)))
    else:
        sol_z = np.zeros((0,), dtype=np.float32)
        sol_q = np.zeros((0,), dtype=np.float32)
        sol_u = np.zeros((0,), dtype=np.float32)

    np.savez(
        out_path,
        # identity
        scene="S2",
        step=np.asarray(int(step), dtype=np.int32),
        seed=np.asarray(int(getattr(args, "seed", 0)), dtype=np.int32),
        mode=str(getattr(args, "mode", "")),
        variant=str(variant),

        # axis conventions (avoid confusion in 3D plotting)
        occupancy_axes="xyz",
        entropy_axes="zyx",

        # scene
        scene_meta_json=scene_meta_json,
        occupancy_xyz=occ,
        obstacles_aabb=obs_aabb,

        # snapshot inputs
        robot_pos0=robot_pos0,
        frontier_entropy_zyx=ent_zyx,
        coverage=np.asarray(float(coverage), dtype=np.float32),
        repulsion_grad=rep_grad,

        # tasks
        task_pos=task_pos,
        task_deadline=task_deadline,
        task_priority=task_priority,
        task_cluster_id=task_cluster,

        # candidates
        cand_pad=cand_pad,
        cand_mask=cand_mask,
        cand_sizes=cand_sizes,

        # initial comm
        edges0=edges0,
        cap0=cap0,
        delay0=delay0,
        plr0=plr0,
        is_stale0=stale0,
        hops0=hops0,
        pair_capacity0=pairC0,

        # oracle
        oracle_next_pos=oracle_pos,
        oracle_score=np.asarray(float(oracle_score), dtype=np.float32),

        # output action
        z_pos=z_pos,
        z_pos_snap=z_snap,
        cand_index=cand_idx,

        # comm after action (snapped)
        edges1=edges1,
        hops1=hops1,
        pair_capacity1=pairC1,

        # objective
        obj_admm_snap_json=json.dumps(obj_snap, ensure_ascii=False),
        obj_stay_put_json=json.dumps(obj_stay, ensure_ascii=False),

        # settings
        flags_json=flags_json,
        params_json=params_json,

        # RB trace (optional)
        eta_hist=eta_hist,

        # optional full solution vectors (empty if disabled)
        sol_z=sol_z,
        sol_q=sol_q,
        sol_u=sol_u,
    )

    return out_path


# ----------------------------
# JSON helpers
# ----------------------------
def to_jsonable_flags(flags: FeatureFlags) -> Dict[str, Any]:
    try:
        return asdict(flags)
    except Exception:
        return {k: getattr(flags, k) for k in dir(flags) if not k.startswith("_")}


def to_jsonable_params(params: CadmmParams) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in dir(params):
        if k.startswith("_"):
            continue
        v = getattr(params, k)
        if callable(v):
            continue
        try:
            json.dumps(v)
            out[k] = v
        except Exception:
            pass
    return out


# ----------------------------
# Main
# ----------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outer_steps", type=int, default=1)

    ap.add_argument("--mode", type=str, default="stability", choices=["stability", "strict", "functional"])
    ap.add_argument("--qstep_updates", type=str, default="none", choices=["none", "all"])
    ap.add_argument("--activate_objectives", type=int, default=0)

    ap.add_argument("--k_max", type=int, default=120)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--output_dir", type=str, default=os.path.join("outputs", "scene_s2_v2"))
    ap.add_argument("--plot", action="store_true")

    # Visualization bundle export (for thesis figures)
    ap.add_argument("--save_viz", type=int, default=1, help="Save per-step visualization bundle npz")
    ap.add_argument("--save_full_solution", type=int, default=0, help="Also store sol.z/q/u in viz bundle")
    ap.add_argument("--disable_residual_balancing", action="store_true", help="Ablation: disable residual balancing (eta adaptation)")
    ap.add_argument("--disable_flow_budget_coupling", action="store_true", help="Ablation: disable flow<->budget coupled projection")
    ap.add_argument("--disable_sigma_y_coupling", action="store_true", help="Ablation: disable sigma<->y_hat coupled projection")

    # S2 constants (must match builder unless you intentionally change both)
    ap.add_argument("--comm_threshold", type=float, default=15.0)
    ap.add_argument("--Cmax", type=float, default=16.0)
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--step_len_ugv", type=float, default=1.0)
    ap.add_argument("--step_len_uav", type=float, default=2.0)

    # strict gate options
    ap.add_argument("--allow_hit_kmax", type=int, default=0)
    ap.add_argument("--abs_factor", type=float, default=50.0)

    # functional objective weights (for evaluation only)
    ap.add_argument("--obj_w_ent", type=float, default=1.0)
    ap.add_argument("--obj_w_cap", type=float, default=0.2)
    ap.add_argument("--obj_w_conn_penalty", type=float, default=50.0)

    args = ap.parse_args()

    seed = int(args.seed)
    out_dir_base = str(args.output_dir)
    variant = _variant_tag_from_args(args)
    out_dir = out_dir_base if variant == "full" else os.path.join(out_dir_base, variant)
    os.makedirs(out_dir, exist_ok=True)

    flags = default_flags_s2()
    flags.enable_residual_balancing = True
    flags.enable_over_relax = True
    flags.enable_async_updates = True
    setattr(flags, "qstep_strict_prox", False)

    # Ablation toggles (for thesis figures)
    if bool(getattr(args, "disable_residual_balancing", False)):
        flags.enable_residual_balancing = False
    if bool(getattr(args, "disable_flow_budget_coupling", False)):
        flags.enable_flow_coupled_to_budget = False
    if bool(getattr(args, "disable_sigma_y_coupling", False)):
        flags.enable_sigma_coupled_to_y = False

    if args.qstep_updates == "none":
        flags = disable_qstep_hard_updates(flags)

    params = default_params_s2(
        k_max=int(args.k_max),
        eps=float(args.eps),
        activate_objectives=bool(int(args.activate_objectives)),
    )
    params.eta = 10.0
    params.k_max = 200
    rng = np.random.default_rng(seed)
    warm: Optional[CadmmWarmStart] = None
    window_state = None

    all_steps: List[Dict[str, Any]] = []
    t0 = time.time()

    for step in range(int(args.outer_steps)):
        env = make_scene_s2_problem(seed=seed, step=step, window_state=window_state, params=params, flags=flags)
        window_state = env.window

        snapshot = build_problem_snapshot(env, step=step, params=params, flags=flags, rng=rng, optional_overrides=None)

        # Sanity checks MUST pass in all modes
        sanity_reasons, sanity_stats = check_s2_builder_and_snapshot(
            env, snapshot,
            comm_threshold=float(args.comm_threshold),
            Cmax=float(args.Cmax),
            resolution=float(args.resolution),
            step_len_ugv=float(args.step_len_ugv),
            step_len_uav=float(args.step_len_uav),
        )

        sol, warm, diag, trace = run_inner_solver_with_trace(snapshot, warm_start=warm, rng=rng)
        setattr(diag, "_params", params)  # for strict hit_kmax check
        

        reasons = list(sanity_reasons)
        stats = dict(sanity_stats)

        # evaluate per mode
        if args.mode == "stability":
            r2, s2 = eval_stability(trace, diag)
            reasons += r2
            stats.update({f"stability_{k}": v for k, v in s2.items()})

        elif args.mode == "strict":
            r2, s2 = eval_strict(
                trace, diag,
                eps=float(args.eps),
                abs_factor=float(args.abs_factor),
                allow_hit_kmax=bool(int(args.allow_hit_kmax)),
            )
            reasons += r2
            stats.update({f"strict_{k}": v for k, v in s2.items()})

        else:  # functional
            r2, s2 = eval_functional(
                snapshot, sol, env,
                resolution=float(args.resolution),
                comm_threshold=float(args.comm_threshold),
                Cmax=float(args.Cmax),
                obj_w_ent=float(args.obj_w_ent),
                obj_w_cap=float(args.obj_w_cap),
                obj_w_conn_penalty=float(args.obj_w_conn_penalty),
            )
            reasons += r2
            stats.update({f"functional_{k}": v for k, v in s2.items()})

        passed = (len(reasons) == 0)
        tag = "PASS" if passed else "FAIL"
        K = int(getattr(diag, "iters", len(trace.r_total)))
        print(f"[Scene S2][step={step}] {tag} iters={K} mode={args.mode} qstep_updates={args.qstep_updates} reasons={reasons}")

        # Save NPZ
        npz_path = os.path.join(out_dir, f"results_step{step}.npz")
        proj_keys = ["sigma", "f_hat", "B_hat", "y_hat"]
        proj_pre = np.array([[float(d.get(k, 0.0)) for k in proj_keys] for d in trace.proj_pre_by_block], dtype=np.float32) if trace.proj_pre_by_block else np.zeros((0,4), dtype=np.float32)
        proj_post = np.array([[float(d.get(k, 0.0)) for k in proj_keys] for d in trace.proj_post_by_block], dtype=np.float32) if trace.proj_post_by_block else np.zeros((0,4), dtype=np.float32)
        proj_corr = np.array([[float(d.get(k, 0.0)) for k in proj_keys] for d in trace.proj_corr_by_block], dtype=np.float32) if trace.proj_corr_by_block else np.zeros((0,4), dtype=np.float32)
        # pack per-block residual norms (for plateau attribution)
        r_keys, r_mat = _pack_norm_series(trace.r_by_name, K)
        s_keys, s_mat = _pack_norm_series(trace.s_by_name, K)
        eta_hist = np.asarray(getattr(diag, "eta_hist", [])[:K], dtype=np.float32)

        np.savez(
            npz_path,
            r_total=np.asarray(trace.r_total, dtype=np.float32),
            s_total=np.asarray(trace.s_total, dtype=np.float32),

            # NEW: per-block residuals
            r_block_keys=r_keys,
            r_by_block=r_mat,
            s_block_keys=s_keys,
            s_by_block=s_mat,

            # NEW: eta history (useful for oscillation diagnosis)
            eta_hist=eta_hist,

            proj_keys=np.asarray(proj_keys, dtype=str),
            proj_pre_violation=proj_pre,
            proj_post_violation=proj_post,
            proj_corr_norm=proj_corr,
            budget_violation=np.asarray(trace.budget_violation, dtype=np.float32),
            coupled_pass_sigma=np.asarray(trace.coupled_pass_sigma, dtype=np.int8),
            coupled_pass_f_hat=np.asarray(trace.coupled_pass_f_hat, dtype=np.int8),
        )

        if bool(int(getattr(args, "save_viz", 1))):
            _ = save_viz_bundle(
                out_dir,
                step=int(step),
                args=args,
                env=env,
                snapshot=snapshot,
                sol=sol,
                diag=diag,
                params=params,
                flags=flags,
                resolution=float(args.resolution),
                comm_threshold=float(args.comm_threshold),
                Cmax=float(args.Cmax),
            )

        if bool(args.plot):
            plot_curves(out_dir, trace, diag)

        def _topk(d: Dict[str, float], k: int = 6) -> List[Tuple[str, float]]:
            items = [(kk, float(vv)) for kk, vv in d.items()]
            items.sort(key=lambda x: -x[1])
            return items[:k]

        r_last = trace.r_by_name[K - 1] if (K > 0 and len(trace.r_by_name) >= K) else {}
        s_last = trace.s_by_name[K - 1] if (K > 0 and len(trace.s_by_name) >= K) else {}

        stats["r_last_top6"] = _topk(r_last, 6)
        stats["s_last_top6"] = _topk(s_last, 6)

        all_steps.append(
            {
                "step": int(step),
                "passed": bool(passed),
                "mode": str(args.mode),
                "qstep_updates": str(args.qstep_updates),
                "reasons": list(reasons),
                "iters": int(K),
                "stats": dict(stats),
            }
        )

    elapsed = time.time() - t0
    overall_pass = bool(all(s["passed"] for s in all_steps))
    overall_tag = "PASS" if overall_pass else "FAIL"

    summary = {
        "scene": "S2",
        "overall": overall_tag,
        "variant": str(variant),
        "seed": int(seed),
        "outer_steps": int(args.outer_steps),
        "elapsed_sec": float(elapsed),
        "flags": to_jsonable_flags(flags),
        "params": to_jsonable_params(params),
        "steps": all_steps,
    }

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[Scene S2] Overall {overall_tag}. Output: {out_dir}")
    return 0 if overall_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())