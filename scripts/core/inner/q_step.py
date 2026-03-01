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

import sys
from typing import Optional, Dict, Tuple
import numpy as np

from scripts.core.data import CadmmProblem
from scripts.core.inner.blocks import BlockRegistry, decode_pos, set_block, get_block
import scripts.core.inner.coverage_metrics as coverage_metrics
import scripts.core.inner.qos_metrics as qos_metrics
from scripts.utils.hops import capacity_for_rid_candidates_minlen_src

# ---------------------------------------------------------------------------
# Cost terms (Stage A)
# ---------------------------------------------------------------------------
def point_to_aabbs_distance(
    c: np.ndarray, obstacles_lo: np.ndarray, obstacles_hi: np.ndarray
) -> np.ndarray:
    """
    c: (3,)
    obstacles_lo/hi: (K,3)
    return d: (K,) distances (>=0), inside box gives 0
    """
    c = np.asarray(c, dtype=np.float32).reshape(1, 3)
    lo = np.asarray(obstacles_lo, dtype=np.float32)
    hi = np.asarray(obstacles_hi, dtype=np.float32)
    # dx = max(lo - c, 0, c - hi) per axis
    delta = np.maximum(lo - c, 0.0) + np.maximum(c - hi, 0.0)
    # NOTE: the above "plus" works because only one side is positive per axis.
    # If you prefer explicit:
    # delta = np.maximum(np.maximum(lo - c, 0.0), c - hi)
    d = np.linalg.norm(delta, axis=1)
    return d


# ---- cached offsets for speed ----
_GAIN_GRID_CACHE: Dict[Tuple[int, int, int, int, float], np.ndarray] = {}  # key -> gain_grid

_OFFSETS_CACHE_3D: Dict[int, np.ndarray] = {}

def _ball_offsets(rv: int) -> np.ndarray:
    rv = int(rv)
    if rv in _OFFSETS_CACHE_3D:
        return _OFFSETS_CACHE_3D[rv]
    pts = []
    r2 = rv * rv
    for dz in range(-rv, rv + 1):
        for dy in range(-rv, rv + 1):
            for dx in range(-rv, rv + 1):
                if dx*dx + dy*dy + dz*dz <= r2:
                    pts.append((dz, dy, dx))
    arr = np.asarray(pts, dtype=np.int32)
    _OFFSETS_CACHE_3D[rv] = arr
    return arr

def _compute_gain_grid_zyx(ent_zyx: np.ndarray, *, sense_radius: float = 5.0, resolution: float = 1.0) -> np.ndarray:
    """
    ent_zyx: (D,H,W) float32 in [0,1]
    return gain_grid_zyx: (D,H,W) mean entropy in sensing ball
    """
    ent = np.asarray(ent_zyx, dtype=np.float32)
    assert ent.ndim == 3
    D, H, W = ent.shape
    rv = int(np.ceil(float(sense_radius) / float(resolution)))
    offs = _ball_offsets(rv)

    # pad to avoid bounds checks inside loop
    pad = rv
    ent_pad = np.pad(ent, ((pad,pad),(pad,pad),(pad,pad)), mode="constant", constant_values=0.0)
    acc = np.zeros((D, H, W), dtype=np.float32)

    # each offset is a pure slice add (fast)
    for dz, dy, dx in offs:
        z0 = pad + dz
        y0 = pad + dy
        x0 = pad + dx
        acc += ent_pad[z0:z0+D, y0:y0+H, x0:x0+W]

    gain = acc / float(len(offs))
    return gain.astype(np.float32, copy=False)

def get_gain_grid(ent_zyx: np.ndarray, *, sense_radius: float = 5.0, resolution: float = 1.0) -> np.ndarray:
    # cache by (data_ptr, D,H,W, sense_radius, resolution)
    ent = np.asarray(ent_zyx, dtype=np.float32)
    D, H, W = int(ent.shape[0]), int(ent.shape[1]), int(ent.shape[2])
    ptr = int(ent.__array_interface__["data"][0])
    key = (ptr, D, H, W, float(sense_radius), float(resolution))
    gg = _GAIN_GRID_CACHE.get(key, None)
    if gg is None or gg.shape != ent.shape:
        gg = _compute_gain_grid_zyx(ent, sense_radius=sense_radius, resolution=resolution)
        _GAIN_GRID_CACHE[key] = gg
    return gg


def cost_admm_quadratic(cand: np.ndarray, target: np.ndarray, eta: float) -> float:
    """0.5 * eta * ||cand - target||^2."""
    d = cand - target
    return 0.5 * float(eta) * float(np.dot(d, d))


def cost_move(cand: np.ndarray, current_pos: np.ndarray, move_w: float) -> float:
    """move_w * ||cand - current_pos||^2."""
    d = cand - current_pos
    return float(move_w) * float(np.dot(d, d))


# def cost_explore_from_frontier_entropy(
#     cand: np.ndarray,
#     frontier_entropy: np.ndarray,
#     new_w: float,
# ) -> float:
#     """- new_w * entropy_at(cand) with safe bounds handling.

#     Coordinate convention:
#       - cand = (x, y) or (x, y, z)
#       - frontier_entropy indexed as [z, y, x]

#     Out-of-bounds -> entropy = 0.
#     """
#     ent = np.asarray(frontier_entropy)
#     if ent.ndim == 3:
#         D, H, W = int(ent.shape[0]), int(ent.shape[1]), int(ent.shape[2])
#         x = int(np.round(float(cand[0])))
#         y = int(np.round(float(cand[1])))
#         z = int(np.round(float(cand[2])))
#         if (x < 0) or (x >= W) or (y < 0) or (y >= H) or (z < 0) or (z >= D):
#             e = 0.0
#         else:
#             e = float(ent[z, y, x])
#         return -float(new_w) * e
#     elif ent.ndim == 2:
#         H, W = int(ent.shape[0]), int(ent.shape[1])
#         x = int(np.round(float(cand[0])))
#         y = int(np.round(float(cand[1])))
#         if (x < 0) or (x >= W) or (y < 0) or (y >= H):
#             e = 0.0
#         else:
#             e = float(ent[y, x])
#         return -float(new_w) * e
#     else:
#         return 0.0
def cost_explore_from_frontier_entropy(
    cand: np.ndarray,
    frontier_entropy: np.ndarray,
    new_w: float,
    *,
    sense_radius: float = 5.0,
    resolution: float = 1.0,
) -> float:
    ent = np.asarray(frontier_entropy, dtype=np.float32)
    if ent.ndim != 3:
        return 0.0

    gain_grid = get_gain_grid(ent, sense_radius=sense_radius, resolution=resolution)

    D, H, W = int(ent.shape[0]), int(ent.shape[1]), int(ent.shape[2])
    x = float(cand[0]); y = float(cand[1]); z = float(cand[2] if cand.size >= 3 else 0.0)

    ix = int(np.clip(np.floor(x / resolution), 0, W - 1))
    iy = int(np.clip(np.floor(y / resolution), 0, H - 1))
    iz = int(np.clip(np.floor(z / resolution), 0, D - 1))

    g = float(gain_grid[iz, iy, ix])
    if not np.isfinite(g):
        g = 0.0
    return -float(new_w) * g

def cost_explore_from_grid_gain(
    cand: np.ndarray,
    grid_gain: np.ndarray,
    new_w: float,
    *, 
    resolution: float = 1.0
) -> float:
    gg = np.asarray(grid_gain,dtype=np.float32)
    if gg.ndim != 3: return 0.0
    D, H, W = int(gg.shape[0]), int(gg.shape[1]), int(gg.shape[2])
    x = float(cand[0]); y = float(cand[1]); z = float(cand[2] if cand.size >= 3 else 0.0)
    ix = int(np.clip(np.floor(x / resolution), 0, W - 1))
    iy = int(np.clip(np.floor(y / resolution), 0, H - 1))
    iz = int(np.clip(np.floor(z / resolution), 0, D - 1))
    g = float(gg[iz, iy, ix])
    if not np.isfinite(g):
        g = 0.0
    return -float(new_w) * g


def cost_task_distance(
    cand: np.ndarray,
    task_pos: np.ndarray,
    priority: np.ndarray,
    deadline: np.ndarray,
    urgency_w: float,
    coord_dim: int
) -> float:
    """urgency_w * min_j (urgency_j * ||cand - task_pos_j||).

    urgency_j = priority_j / max(deadline_j, 1.0)
    If no tasks -> 0.
    """
    tp = np.asarray(task_pos, dtype=np.float32).reshape(-1, coord_dim)
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

def cost_task_direction(
    cand: np.ndarray,
    current_pos: np.ndarray,
    task_pos: np.ndarray,
    priority: np.ndarray,
    deadline: np.ndarray,
    urgency_w: float,
    coord_dim: float,
    mode: str = "top1",  # "top1" | "weighted"
    eps: float = 1e-6
) -> float:
    """
    Direction-based task cost:
      - compute move direction (current -> cand)
      - compute task direction (current -> task)
      - angle in [0,pi], normalized to [0,1] by /pi
      - deadline weight w_ddl = 1/max(deadline,1)
      - optional priority normalization to keep magnitude stable
    Returns in [0, urgency_w] (approximately).
    """
    tp = np.ndarray(task_pos, dtype=np.float32)
    if tp.ndim != 2 or tp.shape[0] == 0: return 0.0

    cand = np.asarray(cand, dtype=np.float32).reshape(-1)
    cur = np.asarray(current_pos, dtype=np.float32).reshape(-1)

    # Handle 2D task_pos while coord_dim = 3 pad z
    if tp.shape[1] == 2 and coord_dim == 3:
        z0 = float(cur[2]) if cur.size >= 3 else 0.0
        tp = np.concatenate([tp, np.full((tp.shape[0], 1), z0, dtype=np.float32)], axis=1)
    
    tp = tp.reshape(-1, coord_dim)
    pr = np.asarray(priority, dtype=np.float32).reshape(-1)
    dl = np.asarray(deadline, dtype=np.float32).reshape(-1)

    T = int(min(tp.shape[0], pr.size, dl.size))
    if T <= 0: return 0.0
    tp = tp[:T]
    pr = pr[:T]
    dl = dl[:T]

    # Move vector
    vm = (cand[:coord_dim] - cur[:coord_dim]).astype(np.float32)
    nvm = float(np.linalg.norm(vm))
    if nvm <= eps:
        return float(urgency_w) * 1.0
    
    # Task vectors from current_pos
    vt = (tp - cur[:coord_dim][None, :]).astype(np.float32)
    nvt = np.linalg.norm(vt, axis=1) + eps

    # Angle via dot-product -> [0,pi]
    dot = (vt @ vm)  # (T,)
    cos = dot / (nvt * nvm)
    cos = np.clip(cos, -1.0, 1.0)
    ang = np.arccos(cos)
    ang_n = ang / np.pi

    # Deadline weight: 1/max(ddl,1) in (0,1]
    w_ddl = 1.0 / np.maximum(dl, 1.0)

    # Priority normalization to keep magnitude stable (optional but recommended)
    pr_max = float(np.max(pr)) if pr.size > 0 else 1.0
    pr_n = pr / max(pr_max, eps)

    # Combined urgency weight in (0, 1]
    w = w_ddl * pr_n

    mode = str(mode).lower()
    if mode == "weighted":
        ww = w / (float(np.sum(w)) + eps)
        val = float(np.sum(ww * ang_n))
    else:
        j = int(np.argmax(w))
        val = float(w[j] * ang_n[j])
    
    return float(urgency_w) * float(np.clip(val, 0.0, 1.0))


# def cost_repulsion(
#     cand: np.ndarray,
#     current_pos: np.ndarray,
#     repulsion_grad_i: np.ndarray,
#     rep_w: float,
#     coord_dim: int
# ) -> float:
#     """rep_w * dot(repulsion_grad[rid], cand-current_pos)."""
#     grad = np.asarray(repulsion_grad_i, dtype=np.float32).reshape(coord_dim)
#     delta = cand - current_pos
#     return float(rep_w) * float(np.dot(grad, delta))
def cost_repulsion(
    cand: np.ndarray, 
    current_pos: np.ndarray,
    robot_pos: np.ndarray, 
    sigma_rep: float,
    rep_w: float
) -> float:
    """
    Repulsion cost to prevent robots from getting too close.

    This implements a nearest-neighbor penalty:
        c_rep = rep_w * phi(d_min)
    where d_min is the distance from `cand` to the nearest other robot.

    Two modes are supported via `sigma_rep`:
      1) Quadratic hinge (recommended): sigma_rep is a dict or tuple containing d0
         - dict: {"mode":"quad", "d0": float}
         - tuple/list: ("quad", d0)
         phi(d) = [max(0, (d0 - d)/d0)]^2  in [0,1]

      2) Exponential (normalized): sigma_rep provides ("exp", d0, sigma)
         phi(d) = (exp(-d/sigma) - exp(-d0/sigma)) / (1 - exp(-d0/sigma)) for d<d0 else 0
         so phi(0)=1, phi(d0)=0, phi in [0,1]

    Parameters
    ----------
    cand : (2,) or (3,)
        Candidate next position for robot i.
    current_pos : (2,) or (3,)
        Current position of robot i (used to identify self row in rebot_pos).
    rebot_pos : (N,2) or (N,3)
        Positions of all robots at current step (includes self).
    sigma_rep :
        Repulsion configuration (see above). You can also pass a single float as d0 (quad mode).
    rep_w : float
        Weight applied to the repulsion cost.

    Returns
    -------
    float
        Repulsion cost (>=0). Typically in [0, rep_w].
    """
    cand = np.asarray(cand, dtype=np.float32).reshape(-1)
    cur = np.asarray(current_pos, dtype=np.float32).reshape(-1)
    pos = np.asarray(robot_pos, dtype=np.float32)

    if pos.ndim != 2 or pos.shape[0] <= 1: return 0.0

    # Make sure dimensions match
    D = int(pos.shape[1])
    if cand.size != D:
        if cand.size == 2 and D == 3:
            cand = np.array([cand[0], cand[1], 0.0], dtype=np.float32)
        else:
            raise ValueError(f"cand dim {cand.size} != robot_pos dim {D}")
    if cur.size != D:
        if cur.size == 2 and D == 3:
            cur = np.array([cur[0], cur[1], 0.0], dtype=np.float32)
        else:
            raise ValueError(f"curren_pos dim {cur.size} != robot_pos dim {D}")
    
    rep_w = float(rep_w)
    if rep_w < 0.0: return 0.0

    # Identify self row by cloest match to current_pos
    dif = pos - cur.reshape(1, D)
    dcur = np.linalg.norm(dif, axis=1)
    self_idx = int(np.argmin(dcur))

    # Compute nearest-neighbor distance from cand to other robots
    dif_c = pos - cand.reshape(1, D)
    dist = np.linalg.norm(dif_c, axis=1)
    dist[self_idx] = np.inf
    d_min = float(np.min(dist)) if dist.size > 0 else np.inf
    if not np.isfinite(d_min): return 0.0

    mode = "quad"
    d0 = None
    sig = None

    if isinstance(sigma_rep, (int, float, np.floating)):
        mode = "quad"
        d0 = float(sigma_rep)
    elif isinstance(sigma_rep, dict):
        mode = str(sigma_rep.get("mode", "quad"))
        d0 = float(sigma_rep.get("d0", 0.0))
        sig = sigma_rep.get("sigma", None)
        sig = float(sig) if sig is not None else None
    elif isinstance(sigma_rep, (tuple, list)) and len(sigma_rep) >= 2:
        mode = str(sigma_rep[0])
        d0 = float(sigma_rep[1])
        if len(sigma_rep) >= 3:
            sig = float(sigma_rep[2])
    else:
        # fallback: try to read attribute .d0 / .sigma / .mode
        mode = str(getattr(sigma_rep, "mode", "quad"))
        d0 = float(getattr(sigma_rep, "d0", 0.0))
        sig = getattr(sigma_rep, "sigma", None)
        sig = float(sig) if sig is not None else None

    d0 = float(d0 if d0 is not None else 0.0)
    if d0 <= 1e-6:
        return 0.0

    # Compute phi(d_min) in [0,1]
    if mode.lower() in ("quad", "quadratic", "hinge"):
        # phi = ((d0 - d)/d0)^2 for d<d0 else 0
        t = (d0 - d_min) / d0
        if t <= 0.0:
            phi = 0.0
        else:
            phi = float(t * t)
            if phi > 1.0:
                phi = 1.0

    elif mode.lower() in ("exp", "exponential"):
        # normalized exponential: phi(0)=1, phi(d0)=0, phi in [0,1]
        if sig is None or sig <= 1e-6:
            sig = 0.3 * d0  # sensible default
        if d_min >= d0:
            phi = 0.0
        else:
            a = np.exp(-d_min / sig)
            b = np.exp(-d0 / sig)
            denom = float(1.0 - b)
            if denom <= 1e-9:
                phi = 0.0
            else:
                phi = float((a - b) / denom)
                phi = float(np.clip(phi, 0.0, 1.0))
    else:
        # Unknown mode -> default to quadratic hinge
        t = (d0 - d_min) / d0
        phi = float(t * t) if t > 0.0 else 0.0

    return float(rep_w) * float(phi)

def cost_clearance(
    cand: np.ndarray,
    obstacles_lo: np.ndarray,
    obstacles_hi: np.ndarray,
    rho: float | np.ndarray,
    clr_w: float,
    *,
    mode: str = "max" # | "general"
) -> float:
    """
    c_clr(c) = clr_w * max_k psi(d_k(c); rho_k), psi in [0, 1]
    if obstacles empty -> 0
    """
    lo = np.asarray(obstacles_lo, dtype=np.float32)
    hi = np.asarray(obstacles_hi, dtype=np.float32)
    if lo.ndim != 2 or lo.shape[0] == 0:
        return 0.0
    
    c = np.asarray(cand, dtype=np.float32).reshape(-1)
    if c.size == 2:
        c = np.array([c[0], c[1], 0.0], dtype=np.float32)
    else:
        c = c[:3]

    rho_arr = np.asarray(rho, dtype=np.float32)
    if rho_arr.ndim == 0:
        rho_arr = np.full((lo.shape[0],), float(rho_arr), dtype=np.float32)
    else:
        rho_arr = rho_arr.reshape(-1)
        if rho_arr.shape[0] != lo.shape[0]:
            raise ValueError(f"rho_k shape {rho_arr.shape} != num obstacles {lo.shape[0]}")
    
    rho_arr = np.maximum(rho_arr, 1e-6)
    d = point_to_aabbs_distance(c, lo, hi)

    if mode == "nearest" and np.allclose(rho_arr, rho_arr[0]):
        dmin = float(np.min(d))
        r = float(rho_arr[0])
        t = max(0.0, 1.0 - dmin / r)
        psi = min(1.0, t * t)
        return float(clr_w) * float(psi)

    t = 1.0 - (d / rho_arr)
    t = np.clip(t, 0.0, 1.0)
    psi_all = t * t
    psi = float(np.max(psi_all))
    return float(clr_w) * psi


def pos_candidate_score(
    *,
    rid: int,      # robot id
    cand: np.ndarray,
    robot_pos: np.ndarray,
    target: np.ndarray,
    current_pos: np.ndarray,
    problem: CadmmProblem,
    eta: float,
    idx: int,       # candidate id
    caps: np.ndarray
) -> float:
    """Composable total cost for selecting a position candidate (Stage A)."""
    flags = getattr(problem, "flags", None)
    params = getattr(problem, "params", None)

    total = 0.0

    # ADMM quadratic term (default ON)
    if bool(getattr(flags, "enable_qstep_admm_term", True)):
        # print(f"ADMM cost: {cost_admm_quadratic(cand, target, eta)}")
        total += cost_admm_quadratic(cand, target, eta)

    # move
    if bool(getattr(flags, "enable_qstep_cost_move", False)):
        move_w = float(getattr(params, "move_w", 0.0))
        move_cost = cost_move(cand, current_pos, move_w)
        # print(f"Move cost: {move_cost}")
        total += move_cost

    # explore
    if bool(getattr(flags, "enable_qstep_cost_explore", False)):
        new_w = float(getattr(params, "new_w", 0.0))
        # print(f"Explore cost: {cost_explore_from_frontier_entropy(cand, problem.frontier_entropy, new_w)}")
        grid_gain = getattr(problem, "grid_gain", None)
        res = getattr(problem, "resolution", 1.0)
        if grid_gain is not None:
            total += cost_explore_from_grid_gain(cand, grid_gain=grid_gain, new_w=new_w, resolution=res)
        else:
            total += cost_explore_from_frontier_entropy(cand, problem.frontier_entropy, new_w)

    # obstacle clearance
    if bool(getattr(flags, "enable_qstep_cost_obstacles", False)):
        obst_w = float(getattr(params, "obst_w", 0.0))
        rho = float(getattr(params, "rho", 0.0))
        obst_cost = cost_clearance(
            cand=cand, 
            obstacles_lo=problem.obstacles_lo,
            obstacles_hi=problem.obstacles_hi,
            rho=rho,
            clr_w=obst_w
        )
        total += obst_cost

    # task
    if bool(getattr(flags, "enable_qstep_cost_task", False)):
        urgency_w = float(getattr(params, "urgency_w", 0.0))
        task = getattr(problem, "task", None)
        if task is not None:
            # task_cost = cost_task_distance(
            #     cand,
            #     getattr(task, "task_pos", np.zeros((0, problem.coord_dim), dtype=np.float32)),
            #     getattr(task, "priority", np.zeros((0,), dtype=np.float32)),
            #     getattr(task, "deadline", np.zeros((0,), dtype=np.float32)),
            #     urgency_w,
            #     problem.coord_dim
            # )
            task_cost = cost_task_direction(
                cand=cand,
                current_pos=current_pos,
                task_pos=getattr(task, "task_pos", np.zeros((0, problem.coord_dim), dtype=np.float32)),
                priority=getattr(task, "priority", np.zeros((0,), dtype=np.float32)),
                deadline=getattr(task, "deadline", np.zeros((0,), dtype=np.float32)),
                urgency_w=urgency_w,
                coord_dim=problem.coord_dim,
                mode=str(getattr(params, "task_mode", "top1")),  # "top1" or "weighted"
            )
            # print(f"Task cost: {task_cost}")
            total += task_cost
            # sys.exit()

    # repulsion
    if bool(getattr(flags, "enable_qstep_cost_repulsion", False)):
        rep_w = float(getattr(params, "rep_w", 0.0))
        rep_grad = np.asarray(
            getattr(problem, "repulsion_grad", np.zeros((problem.N, problem.coord_dim), dtype=np.float32)),
            dtype=np.float32,
        )
        if rep_grad.ndim == 2 and rep_grad.shape[0] > rid and rep_grad.shape[1] >= problem.coord_dim:
            # print(f"Repulsion cost: {cost_repulsion(cand, current_pos, rep_grad[rid], rep_w, problem.coord_dim)}")
            rep_cost = cost_repulsion(
                cand=cand, 
                current_pos=current_pos, 
                robot_pos=robot_pos, 
                sigma_rep={"mode": "quad", "d0": 3.0},
                rep_w=rep_w
            )
            total += rep_cost  # cost_repulsion(cand, current_pos, rep_grad[rid], rep_w, problem.coord_dim)
    
    # qos connectivity
    if bool(getattr(flags, "enable_qstep_cost_qos_pos", False)):
        try: 
            # qos_cost = float(
            #     qos_metrics.connectivity_score_from_positions(
            #         cand,
            #         cands,
            #         getattr(problem, "robot_pos", np.zeros((0,problem.coord_dim), dtype=np.float32)),
            #         params, problem.coord_dim
            #     )
            # )
            c_low = float(getattr(problem, "low_capacity",2.0))
            qos_w = float(getattr(params, "qos_w", 0.0))
            qos_cost = float(
                qos_metrics.connectivity_score_capacity(
                    rid=rid, cand_idx=idx, caps_to=caps, c_low=c_low, qos_w=qos_w
                )
            )
            # print(f"QoS cost: {qos_cost}")
            total += qos_cost
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

    # # Baseline: copy z (keeps other blocks consistent with current pipeline)
    # q_i_new = np.asarray(z, dtype=np.float32).copy()
    # flags = getattr(problem, "flags", None)

    # new added at 20260227 by JiXX
    flags = getattr(problem, "flags", None)
    # Strict ADMM mode: q-step is proximal for f_i(q) = 0 in scaled form
    # q_i := argmin (eta / 2) || q - z + u_i || ^ 2 => q_i = z - u_i
    strict_prox = bool(getattr(flags, "qstep_strict_prox", False))
    if strict_prox:
        q_i_new = (np.asarray(z, dtype=np.float32) - np.asarray(u_i, dtype=np.float32)).copy()
        return q_i_new
    
    # Default (heuristic) mode: copy z then optionally update blocks
    q_i_new = np.asarray(z, dtype=np.float32).copy()

    params = getattr(problem, "params", None)
    coord_dim = problem.coord_dim
    # Default best position for downstream updates
    current_pos = np.asarray(problem.robot_pos[rid], dtype=np.float32).reshape(coord_dim)
    best = current_pos

    # --- Pos Update ---
    # gate for future: disable pos update without removing pos block
    enable_pos_update = bool(getattr(flags, "enable_qstep_update_pos", True))
    if enable_pos_update and ("pos" in reg.names()):
        try:
            candidates = np.asarray(problem.candidate_moves[rid], dtype=np.float32)
        except Exception:
            candidates = np.zeros((0,coord_dim), dtype=np.float32)
        
        if candidates.shape[0] > 0:
            N = int(getattr(problem, "N", 0)) or int(problem.N)
            try:
                target_all = decode_pos(
                    reg, 
                    np.asarray(z, dtype=np.float32) - np.asarray(u_i, dtype=np.float32),
                    N, problem.coord_dim
                )
            except Exception:
                target_all = decode_pos(reg, np.asarray(z, dtype=np.float32), N, problem.coord_dim)
            target = np.asarray(target_all[rid], dtype=np.float32).reshape(coord_dim)
            eta = float(getattr(params, "eta", 1.0)) if params is not None else 1.0

            caps, hops = capacity_for_rid_candidates_minlen_src(
                rid, candidates, problem.robot_pos, problem.los_max_dist, problem.is_los_fn, problem.C_max
            )

            best_idx: int = 0
            best_cost: Optional[float] = None
            for idx in range(int(candidates.shape[0])):
                cand = candidates[idx]
                c = pos_candidate_score(
                    rid=rid,
                    cand=cand,
                    robot_pos=problem.robot_pos,
                    target=target,
                    current_pos=current_pos,
                    problem=problem,
                    eta=eta,
                    idx=idx,
                    caps=caps
                )
                if best_cost is None:
                    best_cost = c
                    best_idx = idx
                else:
                    if (c < best_cost) or (c == best_cost and idx < best_idx):
                        best_cost = c
                        best_idx = idx
            best = np.asarray(candidates[best_idx], dtype=np.float32).reshape(coord_dim)
            pos_all = np.asarray(decode_pos(reg, np.asarray(z, dtype=np.float32), N, problem.coord_dim), dtype=np.float32)
            pos_all = pos_all.copy()
            # if pos_all.ndim == coord_dim and pos_all.shape[0] > rid:
            if pos_all.ndim == 2 and pos_all.shape[0] > rid and pos_all.shape[1] == coord_dim:
                pos_all[rid] = best
                set_block(reg, q_i_new, "pos", pos_all.reshape(-1))
    
    # --- Other blocks ---
    # 1) y_hat: coverage preference
    y_hat_pref: Optional[np.ndarray] = None
    y_hat_final: Optional[np.ndarray] = None
    if ("y_hat" in reg.names()) and bool(getattr(flags, "enable_qstep_update_y_hat", True)):
        try:
            raw_scores = coverage_metrics.coverage_group_scores(problem, rid, best)
            y_hat_pref = coverage_metrics.normalize_to_simplex_nonneg(raw_scores)

            damp = bool(getattr(flags, "enable_qstep_damped_y_hat", False))

            if damp:
                # anchor on ADMM target (z - ui)
                z_minus_u = (np.asarray(z, dtype=np.float32) - np.asarray(u_i, dtype=np.float32))
                y_tgt = np.asarray(get_block(reg, z_minus_u, "y_hat"), dtype=np.float32).reshape(-1)
                beta = float(getattr(params, "y_hat_beta", 1.0))
                if not np.isfinite(beta):
                    beta = 1.0
                beta = float(np.clip(beta, 0.0, 1.0))
                y_mix = (1.0 - beta) * y_tgt + beta * y_hat_pref
                y_hat_final = coverage_metrics.normalize_to_simplex_nonneg(y_mix)
            else:
                y_hat_final = np.asarray(y_hat_pref, dtype=np.float32).reshape(-1)
            set_block(reg, q_i_new, "y_hat", np.asarray(y_hat_final, dtype=np.float32))
        except Exception:
            y_hat_pref = None
            y_hat_final = None
    
    # 2) sigma: coupled to y_hat (damped proximal update)
    if ("sigma" in reg.names()) and bool(getattr(flags, "enable_qstep_update_sigma", True)):
        try:
            damp_s = bool(getattr(flags, "enable_qstep_damped_sigma", False))

            # choose y reference: prefer y_hat_final (after damping), else current block
            if y_hat_final is None and ("y_hat" in reg.names()):
                y_hat_final = np.asarray(get_block(reg, q_i_new, "y_hat"), dtype=np.float32).reshape(-1)
                y_hat_final = coverage_metrics.normalize_to_simplex_nonneg(y_hat_final)

            if y_hat_final is not None:
                sigma_pref = coverage_metrics.sigma_target_from_y_hat(y_hat_final, params, flags)

                if damp_s:
                    z_minus_u = (np.asarray(z, dtype=np.float32) - np.asarray(u_i, dtype=np.float32))
                    sigma_tgt = np.asarray(get_block(reg, z_minus_u, "sigma"), dtype=np.float32).reshape(-1)

                    beta_s = float(getattr(params, "sigma_beta", getattr(params, "y_hat_beta", 1.0)))
                    if not np.isfinite(beta_s):
                        beta_s = 1.0
                    beta_s = float(np.clip(beta_s, 0.0, 1.0))

                    sigma_mix = (1.0 - beta_s) * sigma_tgt + beta_s * np.asarray(sigma_pref, dtype=np.float32).reshape(-1)

                    # optional: enforce coupled lower bound early (z-step will enforce again)
                    sigma_max = float(getattr(params, "sigma_max", 1.0))
                    if bool(getattr(flags, "enable_sigma_coupled_to_y", True)):
                        lo = coverage_metrics.sigma_lower_bound_from_y_hat(y_hat_final, params, flags)
                        sigma_new = np.clip(sigma_mix, lo, sigma_max)
                    else:
                        sigma_new = np.clip(sigma_mix, 0.0, sigma_max)

                    set_block(reg, q_i_new, "sigma", np.asarray(sigma_new, dtype=np.float32))
                else:
                    # legacy behavior (hard overwrite)
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
