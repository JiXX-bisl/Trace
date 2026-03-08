from __future__ import annotations
from dataclasses import dataclass, field
import os
import sys
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import replace
from typing_extensions import Literal 
import numpy as np
from collections import deque
import time

from scripts.core.data import InnerSolution, CadmmProblem, LinkSnapshot, TaskSnapshot, CadmmParams, FeatureFlags, CommWindowState, CadmmWarmStart, CadmmLogConfig, CadmmRunLog, CadmmLog
from scripts.core.cadmm_solver import solve_inner_cadmm
from scripts.core.inner.blocks import make_registry, get_block
from scripts.utils.hops import compute_comm_undirected_minlen

RobotType = Literal["uav", "ugv"]
DemandLevel = Literal["low", "mid", "high"]  # string | array | stream

# --------------------
# Simulation config
# --------------------
# Communication window length (number of env steps per frozen snapshot)
WINDOW_W: int = 10


# --------------------
# Basic Geometry
# --------------------
@dataclass
class AABB:
    """Axis-aligned bounding box obstacle."""
    lo: np.ndarray  # (3,)
    hi: np.ndarray  # (3,)

    def contains(self, p: np.ndarray) -> bool:
        return bool(np.all(p >= self.lo) and np.all(p <= self.hi))
    
def segment_intersects_aabb(p0: np.ndarray, p1: np.ndarray, box) -> bool:
    """Slab method for segment-AABB intersection."""
    p0 = np.asarray(p0, dtype=np.float32)
    p1 = np.asarray(p1, dtype=np.float32)
    d = p1 - p0

    tmin, tmax = 0.0, 1.0
    for i in range(3):
        if abs(float(d[i])) < 1e-9:
            # Segment parallel to slab: must be within slab
            if p0[i] < box.lo[i] or p0[i] > box.hi[i]:
                return False
        else:
            ood = 1.0 / float(d[i])
            t1 = (float(box.lo[i]) - float(p0[i])) * ood
            t2 = (float(box.hi[i]) - float(p0[i])) * ood
            if t1 > t2:
                t1, t2 = t2, t1

            # IMPORTANT: update always
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return False

    return True


def aabb_to_voxel_range_xyz(lo: np.ndarray, hi: np.ndarray, res: float, 
                            nx: int, ny: int, nz: int) -> Tuple[slice, slice, slice]:
    """
    Convert AABB [lo,hi] in meters to voxel index slices for arrays shaped (nx,ny,nz) (x,y,z).
    """
    lo = np.asarray(lo, dtype=np.float32)
    hi = np.asarray(hi, dtype=np.float32)
    res = float(res)

    ix0 = int(np.floor(lo[0] / res)); ix1 = int(np.ceil(hi[0] / res)) - 1
    iy0 = int(np.floor(lo[1] / res)); iy1 = int(np.ceil(hi[1] / res)) - 1
    iz0 = int(np.floor(lo[2] / res)); iz1 = int(np.ceil(hi[2] / res)) - 1

    ix0 = max(0, min(nx - 1, ix0)); ix1 = max(0, min(nx - 1, ix1))
    iy0 = max(0, min(ny - 1, iy0)); iy1 = max(0, min(ny - 1, iy1))
    iz0 = max(0, min(nz - 1, iz0)); iz1 = max(0, min(nz - 1, iz1))

    return slice(ix0, ix1 + 1), slice(iy0, iy1 + 1), slice(iz0, iz1 + 1)

def is_obstacle_revealed_from_grid_xyz(
    *,
    obs_count_xyz: np.ndarray,    # (nx,ny,nz)
    log_odds_xyz: np.ndarray,     # (nx,ny,nz)
    slx: slice, sly: slice, slz: slice,
    p_occ_thres: float = 0.65,
    min_hits: int = 1,
) -> bool:
    """
    Obstacle is considered revealed if within its voxel range there are at least `min_hits`
    voxels that are observed and occupied.
    """
    oc = obs_count_xyz[slx, sly, slz]
    lo = log_odds_xyz[slx, sly, slz]
    seen = oc > 0
    # sigmoid
    p = 1.0 / (1.0 + np.exp(-lo))
    occ = seen & (p > float(p_occ_thres))
    return int(np.count_nonzero(occ)) >= int(min_hits)

def extract_revealed_obstacles_aabb_arrays(
    *,
    obstacles: List,              # List[AABB] with .lo .hi
    obs_count_xyz: np.ndarray,    # (nx,ny,nz)
    log_odds_xyz: np.ndarray,     # (nx,ny,nz)
    res: float,
    p_occ_thres: float = 0.65,
    min_hits: int = 1,
    cached_slices: List[Tuple[slice,slice,slice]] | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[Tuple[slice,slice,slice]]]:
    """
    Returns:
      obstacles_lo: (K,3) float32
      obstacles_hi: (K,3) float32
      idxs: list of obstacle indices that are revealed
      slices: cached_slices (build if None)
    """
    nx, ny, nz = obs_count_xyz.shape
    if cached_slices is None:
        cached_slices = []
        for box in obstacles:
            cached_slices.append(aabb_to_voxel_range_xyz(box.lo, box.hi, res, nx, ny, nz))

    idxs = []
    los = []
    his = []
    for k, box in enumerate(obstacles):
        slx, sly, slz = cached_slices[k]
        if is_obstacle_revealed_from_grid_xyz(
            obs_count_xyz=obs_count_xyz,
            log_odds_xyz=log_odds_xyz,
            slx=slx, sly=sly, slz=slz,
            p_occ_thres=p_occ_thres,
            min_hits=min_hits,
        ):
            idxs.append(k)
            los.append(np.asarray(box.lo, dtype=np.float32))
            his.append(np.asarray(box.hi, dtype=np.float32))

    if len(los) == 0:
        return (np.zeros((0,3), np.float32), np.zeros((0,3), np.float32), [], cached_slices)

    return (np.stack(los, axis=0), np.stack(his, axis=0), idxs, cached_slices)


def extract_frontier_points_xyz(
    disc_xyz: np.ndarray,          # (nx,ny,nz) in {-1,0,1}
    obs_count_xyz: np.ndarray,     # (nx,ny,nz) >=0
    vox_centers: np.ndarray,       # (nx,ny,nz,3) world centers
    *,
    use_26nbr: bool = False,
    max_points: int = 4000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Frontier = seen free voxel that neighbors at least one unknown voxel.
    Returns frontier points as world coords (F,3).
    """
    disc = np.asarray(disc_xyz)
    obs = np.asarray(obs_count_xyz)

    seen = obs > 0
    free_seen = seen & (disc == 0)
    unk = (disc == -1)

    nx, ny, nz = disc.shape
    frontier = np.zeros((nx, ny, nz), dtype=bool)

    # 6-neighborhood (fast and stable)
    def mark_neighbor_unknown(dx, dy, dz):
        xs = slice(max(0, dx), nx + min(0, dx))
        ys = slice(max(0, dy), ny + min(0, dy))
        zs = slice(max(0, dz), nz + min(0, dz))

        xt = slice(max(0, -dx), nx - max(0, dx))
        yt = slice(max(0, -dy), ny - max(0, dy))
        zt = slice(max(0, -dz), nz - max(0, dz))

        frontier[xt, yt, zt] |= free_seen[xt, yt, zt] & unk[xs, ys, zs]

    nbr6 = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    for dx, dy, dz in nbr6:
        mark_neighbor_unknown(dx, dy, dz)

    if use_26nbr:
        # (optional) add diagonal neighbors for thicker frontier
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                for dz in (-1,0,1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    if (dx,dy,dz) in nbr6:
                        continue
                    mark_neighbor_unknown(dx, dy, dz)

    idx = np.argwhere(frontier)
    if idx.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    # downsample to cap cost
    if idx.shape[0] > max_points:
        if rng is None:
            rng = np.random.default_rng(0)
        sel = rng.choice(idx.shape[0], size=max_points, replace=False)
        idx = idx[sel]

    pts = vox_centers[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.float32)  # (F,3)
    return pts

# --------------------
# Candidate move
# --------------------
def point_in_any_aabb(p: np.ndarray, obstacles) -> bool:
    for box in obstacles:
        if np.all(p >= box.lo) and np.all(p <= box.hi):
            return True
    return False

def segment_hits_any_aabb(p0: np.ndarray, p1: np.ndarray, obstacles, seg_fn) -> bool:
    # seg_fn(p0, p1, aabb) -> bool
    for box in obstacles:
        if seg_fn(p0, p1, box):
            return True
    return False

import numpy as np

def build_candidate_moves(
    *,
    pos: np.ndarray,                 # (3,)
    rtype: str,                      # "ugv" or "uav"
    v_max: float,
    dt: float,
    world_size: np.ndarray,          # (3,)
    motion_res_ugv: float = 0.2,
    motion_res_uav: float = 0.5,
    obstacles=None,                  # List[AABB]
    segment_intersects_aabb=None,    # function
    levels=(1.0, 0.5, 0.25),
    max_candidates: int = 81,
    use_segment_check: bool = True,
    bounds_mode: str = "reject",     # "reject" | "clip"
    shape: str = "disk",             # "disk" | "square" (ugv), "ball" | "cube" (uav)
) -> np.ndarray:
    """
    Return candidates: (M,3) float32, world coordinates in meters.
    """
    pos = np.asarray(pos, dtype=np.float32).reshape(3)
    world_size = np.asarray(world_size, dtype=np.float32).reshape(3)
    obstacles = obstacles or []

    r = float(v_max) * float(dt)  # max step distance
    if r <= 1e-6:
        return pos[None, :].astype(np.float32)

    # motion sampling resolution (decoupled from map resolution)
    step0 = float(motion_res_ugv if rtype == "ugv" else motion_res_uav)
    step0 = max(step0, 1e-3)

    # ---- generate offsets (stable unique) ----
    offs = []
    seen = set()

    def add_off(dx: float, dy: float, dz: float):
        # quantize to avoid float key noise
        qx = int(round(dx / step0))
        qy = int(round(dy / step0))
        qz = int(round(dz / step0))
        key = (qx, qy, qz)
        if key in seen:
            return
        seen.add(key)
        offs.append(np.array([qx * step0, qy * step0, qz * step0], dtype=np.float32))

    # always include stay
    add_off(0.0, 0.0, 0.0)

    for a in levels:
        rr = r * float(a)
        rr = max(rr, 1e-6)

        # grid radius for this level
        k = int(np.ceil(rr / step0))
        vals = np.arange(-k, k + 1, dtype=np.int32)

        rr2 = rr * rr + 1e-6

        if rtype == "ugv":
            use_shape = "square" if shape in ("square", "cube") else "disk"
            for iy in vals:
                dy = float(iy) * step0
                for ix in vals:
                    dx = float(ix) * step0
                    if use_shape == "disk":
                        if dx * dx + dy * dy > rr2:
                            continue
                    add_off(dx, dy, 0.0)
        else:
            use_shape = "cube" if shape in ("square", "cube") else "ball"
            for iz in vals:
                dz = float(iz) * step0
                for iy in vals:
                    dy = float(iy) * step0
                    for ix in vals:
                        dx = float(ix) * step0
                        if use_shape == "ball":
                            if dx * dx + dy * dy + dz * dz > rr2:
                                continue
                        add_off(dx, dy, dz)

    # ---- build candidates + filter ----
    cands = []
    p0 = pos.copy()

    for off in offs:
        p = p0 + off
        if rtype == "ugv":
            p[2] = 0.0

        # enforce max step distance (important if shape="square/cube")
        if float(np.linalg.norm(p - p0)) > r + 1e-6:
            continue

        # bounds
        if bounds_mode == "clip":
            p = np.minimum(np.maximum(p, 0.0), world_size)
        else:
            if np.any(p < 0.0) or np.any(p > world_size):
                continue

        # collision: final point
        if point_in_any_aabb(p, obstacles):
            continue

        # optional segment check (can be too strict near obstacles)
        if use_segment_check and segment_intersects_aabb is not None:
            if segment_hits_any_aabb(p0, p, obstacles, segment_intersects_aabb):
                continue

        cands.append(p.astype(np.float32))

    if len(cands) == 0:
        return p0[None, :].astype(np.float32)

    cands = np.stack(cands, axis=0)

    # limit count (deterministic subsample)
    if cands.shape[0] > max_candidates:
        keep = min(12, cands.shape[0])
        rest = cands.shape[0] - keep
        need = max_candidates - keep
        if need <= 0:
            cands = cands[:max_candidates]
        else:
            idx = np.linspace(0, rest - 1, num=need, dtype=int)
            cands = np.concatenate([cands[:keep], cands[keep:][idx]], axis=0)

    return cands
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


# --------------------
# Entites
# --------------------
@dataclass
class Robot:
    rid: int
    rtype: RobotType
    pos: np.ndarray   # (3, ): (x, y, z)
    v_max: float
    sense_r: float = 5.0
    traj: List[np.ndarray] = field(default_factory=list)

    def push_traj(self):
        self.traj.append(self.pos.copy())

@dataclass
class Task:
    tid: int
    pos: np.ndarray               # (3,)
    start_step: int
    deadline_step: int
    demand: DemandLevel
    target: Literal["base", "robot"] = "base"
    target_rid: Optional[int] = None
    done: bool = False

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
# --------------------
# Main Env
# --------------------
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
        time0 = time.time()
        sol, ws, diag = solve_inner_cadmm(problem_snapshot, warm_start, rng)
        time1 = time.time()
        # print(f"[DEBUG] runnning time : {time1 - time0: .2f}")

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

class SceneEnv:
    """
    40m x 40m x 8m 3D world, 3 UAV + 3 UGV, LoS multi-hop comm, voxel map reveal.

    API:
        - reset() -> obs
        - update(next_pos: np.ndarray | None, dt: float = 1.0, ...) -> obs, info
        - render()
    """
    def __init__(
        self,
        *,
        seed: int = 0,
        world_size: Tuple[float, float, float] = (40.0, 40.0, 8.0),
        resolution: float = 1.0,
        base_box: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = ((1.0, 1.0, 0.0), (4.0, 4.0, 2.0)),
        v_ugv: float = 1.0,
        sense_r: float = 5.0,
        los_max_dist: float = 15.0,
        C_max: float = 30.0,
        enable_sense_occlusion: bool = False,
        task_spawn_p: float = 0.05,
        task_deadline_range: Tuple[int, int] = (20, 60),
        enable_vis: bool = False,
    ):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.world_size = np.array(world_size, dtype=np.float32)
        self.res = float(resolution)
        self.base_lo = np.array(base_box[0], dtype=np.float32)
        self.base_hi = np.array(base_box[1], dtype=np.float32)

        self.v_ugv = float(v_ugv)
        self.v_uav = float(2.0 * v_ugv)
        self.sense_r = float(sense_r)
        
        self.los_max_dist = float(los_max_dist)
        self.C_max = float(C_max)

        self.enable_sense_occlusion = bool(enable_sense_occlusion)
        self.task_spawn_p = float(task_spawn_p)
        self.task_deadline_range = task_deadline_range

        self.enable_vis = enable_vis

        # voxel grid
        self.nx = int(np.ceil(self.world_size[0] / self.res))
        self.ny = int(np.ceil(self.world_size[1] / self.res))
        self.nz = int(np.ceil(self.world_size[2] / self.res))

        # map belief
        self.log_odds = np.zeros((self.nx, self.ny, self.nz), dtype=np.float32)
        self.obs_count = np.zeros((self.nx, self.ny, self.nz), dtype=np.float32)

        # Precompute voxel centers
        xs = (np.arange(self.nx) + 0.5) * self.res
        ys = (np.arange(self.ny) + 0.5) * self.res
        zs = (np.arange(self.nz) + 0.5) * self.res
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        self.vox_centers = np.stack([X, Y, Z], axis=-1)  # (nx, ny, nz, 3)

        # obstacles
        self.obstacles: List[AABB] = []
        self.robots: List[Robot] = []
        self.tasks: Dict[int, Task] = {}
        self.step_count = 0
        self._next_tid = 0

        # some parameters
        self.tau = 4.0 #0.75  # confidence parameter for obs_count

        self.reset()
    
    def reset(self):
        self.step_count = 0
        self._next_tid = 0
        self.tasks.clear()

        self.log_odds.fill(0.0)
        self.obs_count.fill(0)

        self._spawn_obstacles(n_obs=10)
        self._spawn_robots()
        for r in self.robots:
            r.traj.clear()
            r.push_traj()

        obs = self._make_obs()
        explore_score = self._compute_explore_score()
        comm = self._compute_comm()
        comm_score = self._compute_comm_score(comm=comm)
        link, _ = self._get_linksnapshot(comm)
        self.window = CommWindowState(W=WINDOW_W, start_step=self.step_count, omega_seed=self.seed + self.step_count, frozen_link=link)
        self.warm = None
        # 1) build explore maps
        frontier_entropy_zyx, grid_gain_zyx  = self._get_ent_grid_gain_snapshot()
        # 2) candidate moves per robot
        candidate_moves = self._get_candidate_moves()
        # 3) revealed obstacles
        obs_lo, obs_hi = self._get_revealed_obsts_snapshot()
        # 4) other basic info
        link, _ = self._get_linksnapshot(comm=comm)        
        # 5) get frontier map
        disc = self.get_discrete_map()
        frontier_points_xyz = extract_frontier_points_xyz(
            disc_xyz=disc,
            obs_count_xyz=self.obs_count,
            vox_centers=self.vox_centers,
            use_26nbr=False,
            max_points=4000,
            rng=self.rng
        )
        return {
            "step": self.step_count,
            "explore_score": float(explore_score),
            "comm_score": float(comm_score),
            "num_tasks": int(len(self.tasks)),
            # --- cadmm ---
            "comm": comm,
            "frontier_entropy_zyx": frontier_entropy_zyx,
            "grid_gain_zyx": grid_gain_zyx,
            "candidate_moves": candidate_moves,
            "obstacle_lo": obs_lo,
            "obstacle_hi": obs_hi,
            "link": link,
            "frontier_points": frontier_points_xyz   
        }
    
    # -------------------
    # Init helper
    # -------------------
    def _spawn_obstacles(self, n_obs: int = 12):
        self.obstacles.clear()

        Z = float(self.world_size[2])

        for _ in range(n_obs):
            for _try in range(200):
                # xy 尺寸随机
                wxy = self.rng.uniform(2.0, 6.0, size=(2,))
                # 高度随机，但必须 <= Z
                h = float(self.rng.uniform(1.0, min(5.0, Z - 0.2)))
                w = np.array([wxy[0], wxy[1], h], dtype=np.float32)

                # lo 的 z 固定为 0：贴地
                lo_xy = self.rng.uniform([0.0, 0.0], self.world_size[:2] - w[:2])
                lo = np.array([lo_xy[0], lo_xy[1], 0.0], dtype=np.float32)
                hi = lo + w

                box = AABB(lo=lo, hi=hi)

                # 避免覆盖 base 区域
                if self._aabb_overlap(box.lo, box.hi, self.base_lo, self.base_hi):
                    continue

                self.obstacles.append(box)
                break
    
    @staticmethod
    def _aabb_overlap(lo1, hi1, lo2, hi2) -> bool:
        return bool(np.all(lo1 <= hi2) and np.all(lo2 <= hi1) and np.all(hi1 >= lo2) and np.all(hi2 >= lo1))
    
    def _spawn_robots(self):
        self.robots.clear()
        N_uav, N_ugv = 6, 0
        rid = 0
        def sample_in_base():
            p = self.rng.uniform(self.base_lo, self.base_hi)
            return p.astype(np.float32)
        
        # UGV: z fixed to 0
        for _ in range(N_ugv):
            p = sample_in_base()
            p[2] = 0.0
            self.robots.append(Robot(rid=rid, rtype="ugv", pos=p, v_max=self.v_ugv, sense_r=self.sense_r))
            rid += 1

        # UAV: allow z in [0, 8]
        for _ in range(N_uav):
            p = sample_in_base()
            p[2] = float(self.rng.uniform(1.0, min(3.0, self.world_size[2] - 0.5)))
            self.robots.append(Robot(rid=rid, rtype="uav", pos=p, v_max=self.v_uav, sense_r=self.sense_r))
            rid += 1
    
    # --------------------
    # Update step
    # --------------------
    def update(self, *, next_pos = None, sol: Optional[InnerSolution] = None, dt = 1.0):
        """
        One simulation step.
        
        - Inputs: next_pos: (N, 3) desired next positions
        - sol: placeholder for InnerSolution; if provided and has 'next_pos', we use it
        """
        self.step_count += 1

        if sol is not None and hasattr(sol, "next_pos"):
            # use unpack to get pos
            desired = np.asarray(sol.next_pos, dtype=np.float32)
        elif next_pos is not None:
            desired = np.asarray(next_pos, dtype=np.float32)
        else:
            # no-op (hold position)
            desired = np.stack([r.pos for r in self.robots], axis=0)
        
        desired = self._apply_motion_constraints(desired, dt=dt)

        # Commit movement
        for i, r in enumerate(self.robots):
            r.pos = desired[i].copy()
            r.push_traj()
        
        # update map via sensing
        self._sense_and_update_map()

        # Spawn / update tasks
        self._task_update()

        # Update communications
        comm = self._compute_comm()

        # Scores / metrics
        explore_score = self._compute_explore_score()
        comm_score = self._compute_comm_score(comm)

        obs = self._make_obs(comm=comm)

        # Build Cadmm Snapshot
        # 1) build explore maps
        frontier_entropy_zyx, grid_gain_zyx  = self._get_ent_grid_gain_snapshot()
        # 2) candidate moves per robot
        candidate_moves = self._get_candidate_moves()
        # 3) revealed obstacles
        obs_lo, obs_hi = self._get_revealed_obsts_snapshot()
        # 4) other basic info
        link, _ = self._get_linksnapshot(comm=comm)
        # 5) get frontier map
        disc = self.get_discrete_map()
        frontier_points_xyz = extract_frontier_points_xyz(
            disc_xyz=disc,
            obs_count_xyz=self.obs_count,
            vox_centers=self.vox_centers,
            use_26nbr=False,
            max_points=4000,
            rng=self.rng
        )


        info = {
            "step": self.step_count,
            "explore_score": float(explore_score),
            "comm_score": float(comm_score),
            "num_tasks": int(len(self.tasks)),     
            # --- cadmm ---
            "comm": comm,
            "frontier_entropy_zyx": frontier_entropy_zyx,
            "grid_gain_zyx": grid_gain_zyx,
            "candidate_moves": candidate_moves,
            "obstacle_lo": obs_lo,
            "obstacle_hi": obs_hi,
            "link": link,
            "frontier_points": frontier_points_xyz             
        }
        return info
    
    def _apply_motion_constraints(self, desired: np.ndarray, dt: float):
        """Clamp by speed, keep in bounds, avoid obstacles"""
        out = desired.copy()
        N = len(self.robots)

        # bounds
        lo = np.array([0.0,0.0,0.0], dtype=np.float32)
        hi = self.world_size.copy()

        for i in range(N):
            r = self.robots[i]
            p0 = r.pos
            p1 = out[i]

            # UGV constraint z = 0
            if r.rtype == "ugv":
                p1[2] = 0.0
            
            # speed clamp
            max_step = r.v_max * float(dt)
            d = p1 - p0
            dist = float(np.linalg.norm(d))
            if dist > max_step and dist > 1e-9:
                p1 = p0 + (d / dist) * max_step
            
            # world bounds
            p1 = np.minimum(np.maximum(p1, lo), hi)

            # obstacle collision, if inside obstacle, reject to previous
            if self._point_in_any_obstacle(p1):
                p1 = p0.copy()
            out[i] = p1.astype(np.float32)
        return out
    
    def _point_in_any_obstacle(self, p: np.ndarray) -> bool:
        for box in self.obstacles:
            if box.contains(p):
                return True
        return False
    
    # -------------------------
    # Sensing / Map update
    # -------------------------
    def _sense_and_update_map(self):
        """
        Update voxel belief in a simple way:
          - within sensor region => mark free/occupied based on whether voxel center is inside any obstacle
          - log-odds update + obs_count for confidence.
        """
        lo_inc_occ = 0.85
        lo_inc_free = -0.40
        lo_min, lo_max = -4.0, 4.0

        centers = self.vox_centers

        for r in self.robots:
            # distance mask
            dp = centers - r.pos.reshape(1, 1, 1, 3)
            dist = np.linalg.norm(dp, axis=-1)

            if r.rtype == "uav":
                mask = dist <= r.sense_r
            else:
                # hemisphere: only voxels with z >= robot_z
                mask = (dist <= r.sense_r) & (centers[..., 2] >= r.pos[2])

            if self.enable_sense_occlusion:
                mask = self._apply_sense_occlusion_mask(r.pos, mask)
            
            idx = np.argwhere(mask)
            for (ix, iy, iz) in idx:
                c = centers[ix, iy, iz]
                occ = self._point_in_any_obstacle(c)
                self.obs_count[ix, iy, iz] += 1
                self.log_odds[ix, iy, iz] += (lo_inc_occ if occ else lo_inc_free)
                self.log_odds[ix, iy, iz] = np.clip(self.log_odds[ix, iy, iz], lo_min, lo_max)

    def _apply_sense_occlusion_mask(self, sensor_pos: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Very slow placeholder. For each voxel in mask, raycast segment vs obstacles.
        Keep for correctness experiments
        """
        new_mask = mask.copy()
        idx = np.argwhere(mask)
        for (ix, iy, iz) in idx:
            c = self.vox_centers[ix, iy, iz]
            blocked = False
            for box in self.obstacles:
                if segment_intersects_aabb(sensor_pos, c, box):
                    blocked = True
                    break
            if blocked:
                new_mask[ix, iy, iz] = False
        return new_mask
    
    def _prob_occ(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.log_odds))
    
    def get_discrete_map(self) -> np.ndarray:
        """
        Return int8 voxel map: -1 unknown, 0 free, 1 occupied
        """
        p = self._prob_occ()
        m = np.full(p.shape, -1, dtype=np.int8)
        seen = self.obs_count > 0
        m[seen & (p < 0.35)] = 0
        m[seen & (p > 0.65)] = 1
        # middle remains -1 as "uncertain"
        return m
    
    # -------------------------
    # Task update
    # -------------------------
    def _task_update(self):
        if self.rng.random() < self.task_spawn_p:
            pos = self._sample_free_position()
            deadline = self.step_count + int(self.rng.integers(self.task_deadline_range[0], self.task_deadline_range[1]))
            demand = self.rng.choice(["low", "mid", "high"])
            t = Task(
                tid=self._next_tid,
                pos=pos,
                start_step=self.step_count,
                deadline_step=deadline,
                demand=demand,
                target="base",
                target_rid=None,
            )
            self.tasks[t.tid] = t
            self._next_tid += 1
        # expire or complete (placeholder competion rule: any robot within sensing range)
        for tid, task in list(self.tasks.items()):
            # print(f"[DEBUG] Task {tid} state: {task.done}")
            if task.done: continue
            if self.step_count > task.deadline_step:
                del self.tasks[tid]
                continue
            for r in self.robots:
                if float(np.linalg.norm(r.pos - task.pos)) <= self.sense_r:
                    task.done = True
                    break
    
    def _sample_free_position(self) -> np.ndarray:
        for _ in range(500):
            p = self.rng.uniform([0, 0, 0], self.world_size).astype(np.float32)
            if not self._point_in_any_obstacle(p):
                return p
        # fallback
        return self.base_lo.copy()
    
    # --------------------
    # Communication
    # --------------------
    def _compute_comm(self):
        """Build LoS adjacency, compute hops, compute end2end capacities"""
        N = len(self.robots)
        pos = np.stack([r.pos for r in self.robots], axis=0)

        out = compute_comm_undirected_minlen(pos=pos, los_max_dist=self.los_max_dist, is_los_fn=self._is_los, C_max=self.C_max)
        return {
            "adj": out["adj"],           # (N,N) bool
            "hops": out["hops"],         # (N,N) int
            "cap": out["cap"],           # (N,N) float
            "pos": out["pos"],           # (N,3)
        }
    
    def _is_los(self, p0: np.ndarray, p1: np.ndarray) -> bool:
        for box in self.obstacles:
            if segment_intersects_aabb(p0, p1, box): return False
        return True
    
    # -------------------------
    # Scoring
    # -------------------------
    def _compute_explore_score(self) -> float:
        """
        "confidence higher => higher score"
        One simple design:
          conf = 1 - exp(-obs_count / tau)
          explore_score = mean(conf)
        """
        tau = 3.0
        conf = 1.0 - np.exp(-self.obs_count.astype(np.float32) / tau)
        return float(conf.mean())
    
    def _compute_comm_score(self, comm: Dict) -> float:
        """
        Try to keep low-capacity exchange between all pairs.
        Define low threshold, measure fraction of pairs meeting it.
        """
        cap = comm["cap"]
        N = cap.shape[0]
        C_low = 0.2  # TODO: Map low/mid/high to numbers later
        ok = 0
        total = 0
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                total += 1
                if cap[i, j] >= C_low:
                    ok += 1
        return float(ok / max(total, 1))

    # -------------------------
    # Observation packing
    # -------------------------
    def _make_obs(self, comm: Optional[Dict] = None) -> Dict:
        if comm is None:
            comm = self._compute_comm()

        obs = {
            "step": self.step_count,
            "robots": [
                {"rid": r.rid, "type": r.rtype, "pos": r.pos.copy(), "v_max": r.v_max, "sense_r": r.sense_r}
                for r in self.robots
            ],
            "map": {
                "resolution": self.res,
                "log_odds": self.log_odds,          # keep as numpy (or copy if you need isolation)
                "obs_count": self.obs_count,
                "discrete": self.get_discrete_map(),
            },
            "comm": comm,
            "tasks": [
                {"tid": t.tid, "pos": t.pos.copy(), "deadline": t.deadline_step, "demand": t.demand}
                for t in self.tasks.values()
            ],
        }
        return obs
    
    # -------------------------
    # Cadmm Params Update
    # -------------------------
    def _set_default_cadmm_params(self, k_max: int, eps: float, activate_objectives: bool) -> CadmmParams:
        # TODO: 使用RL输出各个权重参数
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
            obst_w=0.6,
            rho=1.0,
            role_task_weight=1.0,
            role_relay_weight=1.0,
            role_main_weight=1.0,
            qos_exec_weight=1.0,
            qos_relay_weight=1.0,
            qos_default_weight=1.0,
            qos_break_w=0.0,
            qos_improve_w=0.0,
            qos_degrade_w=0.0,
            urgency_w=0.2 if activate_objectives else 0.0,  # <- task urgency
            rep_w=0.8 if activate_objectives else 0.0,     # <- repulsion
            qos_w=1.0,
            rep_sigma=1.0,
            budget_cache_ema=0.2,
        )
        setattr(p, "eta_stability_ratio", 100.0)
        setattr(p, "root_id_default", 0)
        return p
    
    def _set_default_featureflags(self):
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
            enable_qstep_damped_sigma=True,
            enable_qstep_cost_obstacles=True
        )
        return f
    
    # -------------------------
    # Snapshot Builder
    # -------------------------
    @staticmethod
    def _ball_offsets(rv: int) -> np.ndarray:
        rv = int(rv)
        pts = []
        r2 = rv * rv
        for dz in range(-rv, rv + 1):
            for dy in range(-rv, rv + 1):
                for dx in range(-rv, rv + 1):
                    if dx*dx + dy*dy + dz*dz <= r2:
                        pts.append((dz, dy, dx))
        arr = np.asarray(pts, dtype=np.int32)
        return arr

    def _get_ent_grid_gain_snapshot(self):
        obs_xyz = self.obs_count.astype(np.float32)
        u_xyz = np.exp(-obs_xyz / self.tau)
        frontier_entropy_zyx = np.transpose(u_xyz, (2, 1, 0))  # nz, ny, nx
        frontier_entropy_zyx = np.clip(frontier_entropy_zyx, 0.0, 1.0).astype(np.float32, copy=False)

        D, H, W = int(frontier_entropy_zyx.shape[0]), int(frontier_entropy_zyx.shape[1]), int(frontier_entropy_zyx.shape[2])
        rv = int(np.ceil(float(self.sense_r) / float(self.res)))
        offs = self._ball_offsets(rv)

        # pad to avoid bounds checks inside loop
        pad = rv
        ent_pad = np.pad(frontier_entropy_zyx, ((pad,pad),(pad,pad),(pad,pad)), mode="constant", constant_values=0.0)
        acc = np.zeros((D, H, W), dtype=np.float32)

        # each offset is a pure slice add
        for dz, dy, dx in offs:
            z0 = pad + dz
            y0 = pad + dy
            x0 = pad + dx
            acc += ent_pad[z0:z0+D, y0:y0+H, x0:x0+W]
        gain = acc / float(len(offs))
        return frontier_entropy_zyx, gain.astype(np.float32, copy=False)
    
    def _get_revealed_obsts_snapshot(self):
        self._obs_slices_cache = None
        obs_lo, obs_hi, idxs, self._obs_slices_cache = extract_revealed_obstacles_aabb_arrays(
            obstacles=self.obstacles,
            obs_count_xyz=self.obs_count,
            log_odds_xyz=self.log_odds,
            res=self.res,
            p_occ_thres=0.65,
            min_hits=1,
            cached_slices=self._obs_slices_cache
        )
        return obs_lo, obs_hi
            
    def _get_candidate_moves(self):
        candidate_moves = []
        for r in self.robots:
            cands = build_candidate_moves(
                pos=r.pos,
                rtype=r.rtype,
                v_max=r.v_max,
                dt=1.0,
                world_size=self.world_size,
                obstacles=self.obstacles,
                segment_intersects_aabb=segment_intersects_aabb,
                max_candidates=81 if r.rtype=="ugv" else 121,
                use_segment_check=True, 
                bounds_mode="reject",
                shape="disk" if r.rtype == "ugv" else "ball"
            )
            candidate_moves.append(cands)
        return candidate_moves
    
    def _get_linksnapshot(self, comm):
        adj = np.asarray(comm["adj"], dtype=np.bool_)
        cap_mat = np.asarray(comm["cap"], dtype=np.float32)
        pos = np.asarray(comm["pos"], dtype=np.float32)
        N = int(pos.shape[0])

        # edges:
        edges_list = [(i, j) for i in range(N) for j in range(i + 1, N) if adj[i, j]]
        E = len(edges_list)
        edges = np.asarray(edges_list, dtype=np.int32)

        # capacity:
        capacity = cap_mat[edges[:, 0], edges[:, 1]].astype(np.float32)

        # signap
        signal = capacity.copy()

        # delay
        delay = np.zeros((E,), dtype=np.float32)
        plr = np.zeros((E,), dtype=np.float32)

        is_stale = np.zeros((E,), dtype=np.bool_)

        link = LinkSnapshot(
            robot_ids=list(range(N)),
            edges=edges,
            signal=signal,
            capacity=capacity,
            delay=delay,
            plr=plr,
            is_stale=is_stale,
        )
        return link, E
    
    def _get_tasksnapshot(self):
        task_pos = [t.pos for tid, t in self.tasks.items()]
        deadline = [t.deadline_step for tid, t in self.tasks.items()]
        priority = [1 / (t.deadline_step - t.start_step + 1e-12) for tid, t in self.tasks.items()]
        cluster_id = [0 for tid, t in self.tasks.items()]

        return TaskSnapshot(
            task_pos=task_pos,
            deadline=deadline,
            priority=priority,
            cluster_id=cluster_id
        )
        
    # -------------------------
    # Visualization
    # -------------------------    
    def render(
        self,
        *,
        show_comm: bool = True,
        show_traj: bool = True,
        show_map: bool = True,
        map_show: str = "occ+free",   # "occ" / "occ+unc" / "occ+free"
        free_downsample: int = 3,     # 自由点下采样（越大越稀）
        max_points: int = 6000,       # 防止一次画太多
        save_path: str = "test.png",
        show: bool = False
    ):
        if not self.enable_vis:
            return

        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        tro_rc = {
            "font.family": "Times New Roman",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.2,
            "lines.markersize": 4.0,
            "pdf.fonttype": 42,  # editable text in pdf
            "ps.fonttype": 42,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }

        with mpl.rc_context(tro_rc):
            fig = plt.figure(figsize=(8.0, 5.2))
            ax = fig.add_subplot(111, projection="3d")

            # A) camera / projection (paper-friendly)
            try:
                ax.set_proj_type("ortho")
            except Exception:
                pass
            ax.view_init(elev=22, azim=-55)

            # B) panes + grid (clean, subtle)
            ax.set_facecolor("white")
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                try:
                    axis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
                    axis.pane.set_edgecolor((0.85, 0.85, 0.85, 1.0))
                except Exception:
                    pass

            ax.grid(True)
            # (Matplotlib 3D gridline styling is limited; this covers most versions.)
            try:
                ax.xaxis._axinfo["grid"]["linewidth"] = 0.6
                ax.yaxis._axinfo["grid"]["linewidth"] = 0.6
                ax.zaxis._axinfo["grid"]["linewidth"] = 0.6
                ax.xaxis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 1.0)
                ax.yaxis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 1.0)
                ax.zaxis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 1.0)
            except Exception:
                pass

            # C) Box aspect ratio: H:W:Z = 5:5:1
            # (This is the key "比例调整".)
            try:
                ax.set_box_aspect((5.0, 5.0, 1.0))
            except Exception:
                # Older matplotlib: no reliable 3D aspect control; ignore gracefully.
                pass

            # 1) obstacles (keep your existing draw call)
            for box in self.obstacles:
                self._draw_aabb(ax, box)

            # 2) reveal map
            if show_map:
                disc = self.get_discrete_map()       # -1/0/1
                seen = self.obs_count > 0

                occ = seen & (disc == 1)
                free = seen & (disc == 0)
                unc = seen & (disc == -1)

                C = self.vox_centers  # (nx,ny,nz,3)

                def scatter_mask(mask, *, s=6, alpha=0.15, color=None):
                    idx = np.argwhere(mask)
                    if idx.shape[0] == 0:
                        return
                    if idx.shape[0] > max_points:
                        sel = self.rng.choice(idx.shape[0], size=max_points, replace=False)
                        idx = idx[sel]
                    pts = C[idx[:, 0], idx[:, 1], idx[:, 2]]
                    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=s, alpha=alpha, c=color)

                # TRO-like: background info light, occupied clearer
                scatter_mask(occ, s=18, alpha=0.35, color="0.20")  # darker gray

                if map_show in ("occ+unc", "occ+free"):
                    scatter_mask(unc, s=8, alpha=0.10, color="0.55")  # mid gray

                if map_show == "occ+free":
                    if free_downsample > 1:
                        ds = free_downsample
                        ds_free = np.zeros_like(free, dtype=bool)
                        ds_free[::ds, ::ds, ::ds] = True
                        free = free & ds_free
                    scatter_mask(free, s=4, alpha=0.05, color="0.85")  # very light gray

            # 3) robots + traj
            for r in self.robots:
                p = r.pos
                is_uav = (r.rtype == "uav")
                ax.scatter(
                    p[0], p[1], p[2],
                    marker="^" if is_uav else "o",
                    s=60,
                    edgecolors="0.15",
                    linewidths=0.6,
                )
                if show_traj and len(r.traj) >= 2:
                    tr = np.stack(r.traj, axis=0)
                    ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], linewidth=1.2)

            # 4) comm links
            if show_comm:
                comm = self._compute_comm()
                adj = comm["adj"]
                pos = comm["pos"]
                N = pos.shape[0]
                for i in range(N):
                    for j in range(i + 1, N):
                        if adj[i, j] and adj[j, i]:
                            a, b = pos[i], pos[j]
                            ax.plot(
                                [a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                                linewidth=0.9,
                                alpha=0.65,
                            )

            # 5) tasks
            for t in self.tasks.values():
                ax.scatter(
                    t.pos[0], t.pos[1], t.pos[2],
                    marker="*",
                    s=90,
                    edgecolors="0.15",
                    linewidths=0.6,
                )

            ax.set_xlim(0, self.world_size[0])
            ax.set_ylim(0, self.world_size[1])
            ax.set_zlim(0, self.world_size[2])

            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Z (m)")
            ax.set_title(f"SceneEnv step={self.step_count}")

            plt.tight_layout(pad=0.6)

            # save first (non-blocking)
            if save_path is not None:
                save_dir = os.path.dirname(save_path)
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(save_path, dpi=300)  # 原第1233行代码
                # plt.savefig(save_path, dpi=300)

            # show optional
            if show:
                plt.show()
            else:
                plt.close(fig)

    def _draw_aabb(self, ax, box: AABB):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        lo, hi = box.lo, box.hi
        x0, y0, z0 = lo
        x1, y1, z1 = hi
        # 8 vertices
        v = np.array([
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ], dtype=np.float32)
        faces = [
            [v[0], v[1], v[2], v[3]],
            [v[4], v[5], v[6], v[7]],
            [v[0], v[1], v[5], v[4]],
            [v[2], v[3], v[7], v[6]],
            [v[1], v[2], v[6], v[5]],
            [v[0], v[3], v[7], v[4]],
        ]
        poly = Poly3DCollection(faces, alpha=0.15)
        ax.add_collection3d(poly)


# --------------------
# Algo modes (stable entrypoints)
# --------------------
def make_flags_theory(env: "SceneEnv") -> FeatureFlags:
    """Route-B (theory-aligned) flags."""
    f = env._set_default_featureflags()
    f = replace(
        f,
        enable_theory_mode=True,
        enable_sigma_y_joint_polytope=True,
        theory_y_hat_box_only=True,
        enable_sigma_coupled_to_y=False,
    )
    return f

def make_flags_legacy(env: "SceneEnv") -> FeatureFlags:
    """Legacy flags (pre Route-B), explicitly disabling theory switches."""
    f = env._set_default_featureflags()
    f = replace(
        f,
        enable_theory_mode=False,
        enable_sigma_y_joint_polytope=False,
        theory_y_hat_box_only=False,
    )
    return f

def maybe_refresh_comm_window(env: "SceneEnv", link: LinkSnapshot, *, W: int = WINDOW_W) -> None:
    """Refresh frozen snapshot at the start of each communication window.

    Within a window, env.window.frozen_link stays constant.
    """
    ws = getattr(env, "window", None)
    step = int(getattr(env, "step_count", 0))
    if ws is None:
        env.window = CommWindowState(W=int(W), start_step=step, omega_seed=int(getattr(env, "seed", 0)) + step, frozen_link=link)
        return
    start = int(getattr(ws, "start_step", step))
    wlen = int(getattr(ws, "W", W))
    if step - start >= wlen:
        env.window = CommWindowState(W=int(W), start_step=step, omega_seed=int(getattr(env, "seed", 0)) + step, frozen_link=link)

def build_cadmm_problem(env: SceneEnv, obs):
    # Ensure we have a consistent frozen snapshot per communication window
    maybe_refresh_comm_window(env, obs["link"], W=WINDOW_W)
    frozen_link = env.window.frozen_link if getattr(env, "window", None) is not None else obs["link"]
    edges = np.asarray(getattr(frozen_link, "edges", np.zeros((0, 2), dtype=np.int32)), dtype=np.int32)
    E = int(edges.shape[0])

    flags = make_flags_theory(env)
    M_max = max(int(c.shape[0]) for c in obs["candidate_moves"])

    flags = replace(
        flags,
        enable_theta=True,
        theta_dim=int(M_max),
        enable_qstep_update_theta=True,
        enable_y_avg_assembly=True,
    )
    params = env._set_default_cadmm_params(k_max=120, eps=1e-5, activate_objectives=True)
    setattr(params, "theta_eta_y", 1.0)
    setattr(params, "theta_eta_prior", 0.1)
    setattr(params, "theta_pg_iters", 8)
    setattr(params, "eta_y_avg_scale", 1.0) 

    cadmm_problem = CadmmProblem(
        N=len(env.robots),                   # num of robots
        E=E,                                 # num of frozen LoS edges
        G=2,                                 # num of clusters
        T=len(env.tasks),                    # num of tasks (NOTE: not time-horizon T)
        robots=env.robots,
        robot_pos=np.stack([r.pos for r in env.robots], axis=0).astype(np.float32),
        candidate_moves=obs["candidate_moves"],
        coverage=float(obs["explore_score"]),
        frontier_entropy=obs["frontier_entropy_zyx"],
        grid_gain=obs["grid_gain_zyx"],
        frontier_pts=obs["frontier_points"],

        link=frozen_link,
        task=env._get_tasksnapshot(),

        obstacle_lo=obs["obstacle_lo"],
        obstacle_hi=obs["obstacle_hi"],

        # params=env._set_default_cadmm_params(k_max=120, eps=1e-3, activate_objectives=True),
        params=params,
        flags=flags,
        window=env.window,

        is_los_fn=env._is_los,

        coord_dim=3
    )
    return cadmm_problem


if __name__ == "__main__":
    """sim_new (Route-B / theory-aligned).

    This script is self-contained and hard-codes the algorithm mode via build_cadmm_problem().
    """
    env = SceneEnv(enable_vis=True)
    obs = env.reset()

    # --- rollout-level CADMM log (new/old common) ---
    cadmm_log = CadmmLog(
        config=CadmmLogConfig(
            level="lite",            # switch to 'full' if you also want z/q/u trajectories
            stride=1,
            store_blocks=("pos","f_hat","B_hat","y_hat","sigma","r_hat"),
            store_full_state=False,
            state_dtype="float16",
            save_compressed=True,
        )
    )


    for t in range(100):
        # Build snapshot (also refreshes frozen comm window internally)
        cadmm_problem = build_cadmm_problem(env=env, obs=obs)
        cadmm_problem.log = cadmm_log
        cadmm_problem.rollout_step = int(t)


        # Solve inner CADMM
        time1 = time.time()
        sol, warm, diag, trace = run_inner_solver_with_trace(cadmm_problem, warm_start=env.warm, rng=env.rng)
        time2 = time.time()
        # print(f"time = {time2 - time1: .5f}")
        # if t == 2:
        #     sys.exit()
        env.warm = warm

        # Route-B (theory): prefer solver-decoded next_pos; fall back to z-pos if needed
        desired = None
        if hasattr(sol, "next_pos") and sol.next_pos is not None:
            try:
                desired = np.asarray(sol.next_pos, dtype=np.float32).reshape(cadmm_problem.N, 3)
            except Exception:
                desired = None

        if desired is None:
            # fallback: decode from z
            reg = make_registry(cadmm_problem.N, cadmm_problem.E, cadmm_problem.G, cadmm_problem.flags, T_horizon=int(getattr(cadmm_problem, "T_horizon", 1)))
            z_blocks = reg.unpack(sol.z)
            z_pos = np.asarray(z_blocks.get("pos", cadmm_problem.robot_pos.reshape(-1)), dtype=np.float32).reshape(cadmm_problem.N, 3)
            desired = z_pos

        if getattr(cadmm_problem.flags, "enable_theta", False) and hasattr(sol, "q") and sol.q is not None:
            reg = make_registry(cadmm_problem.N, cadmm_problem.E, cadmm_problem.G, cadmm_problem.flags, T_horizon=int(getattr(cadmm_problem, "T_horizon", 1)))
            M = int(getattr(cadmm_problem.flags, "theta_dim", 0))
            desired = np.zeros((cadmm_problem.N, 3), np.float32)
            for i in range(cadmm_problem.N):
                th_all = np.asarray(get_block(reg, sol.q[i], "theta"), np.float32).reshape(cadmm_problem.N, M)
                th_i = th_all[i]
                cands = np.asarray(cadmm_problem.candidate_moves[i], np.float32)
                Mi = int(cands.shape[0])
                # idx = int(np.argmax(th_i[:Mi]))
                # desired[i] = cands[idx]
                min_move = 0.5
                min_move2 = min_move * min_move

                w = th_i[:Mi].copy()
                w = w / (w.sum() + 1e-12)
                
                d2_curr = np.sum((cands[:Mi] - env.robots[i].pos)**2, axis=1)
                mask = d2_curr >= min_move2

                if np.any(mask):
                    idx = int(np.argmax(w[mask]))
                    chosen = int(np.where(mask)[0][idx])
                else:
                    chosen = int(np.argmax(w))  # 兜底
                desired[i] = cands[chosen]         
        else:
            desired = snap_pos_to_candidates(np.asarray(sol.next_pos, np.float32).reshape(cadmm_problem.N,3),
                                            cadmm_problem.candidate_moves)

        # Enforce discrete-action execution by snapping to candidates (stable)
        desired = snap_pos_to_candidates(desired, cadmm_problem.candidate_moves)
        obs = env.update(next_pos=desired, dt=1.0)
        # Save a frame periodically
        if (t % 10) == 0:
            save_dir = os.path.join("temp", "frames")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{env.step_count:04d}.png")

            env.render(
                show_comm=True,
                show_traj=True,
                show_map=True,
                map_show="occ+free",
                free_downsample=3,
                max_points=6000,
                save_path=save_path,
                show=False,
            )

            finished = 0
            for tid, task in env.tasks.items():
                if task.done:
                    finished += 1
            print("[step {:03d}] explore={:.3f}, comm={:.3f}, tasks_done={}".format(
                env.step_count, float(obs["explore_score"]), float(obs["comm_score"]), int(finished)
            ))

    print("Rollout done. Frames saved in temp/frames")


    # --- save cadmm log once after rollout ---
    import time as _time
    out_dir = f"results/cadmm_log_new_" + _time.strftime("%Y%m%d_%H%M%S")
    cadmm_log.save(out_dir)
    print("Saved CADMM log to:", out_dir)
