from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable
from typing_extensions import Literal 
import numpy as np
from collections import deque

from scripts.core.data import InnerSolution
from scripts.utils.hops import compute_comm_undirected_minlen

RobotType = Literal["uav", "ugv"]
DemandLevel = Literal["low", "mid", "high"]  # string | array | stream

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
    
def segment_intersects_aabb(p0: np.ndarray, p1: np.ndarray, box: AABB) -> bool:
    """Slab method for segment AABB intersections"""
    d = p1 - p0
    tmin, tmax = 0.0, 1.0
    for i in range(3):
        if abs(d[i]) < 1e-9:
            if p0[i] < box.lo[i] or p0[i] > box.hi[i]:
                return False
        else:
            ood = 1.0 / d[i]
            t1 = (box.lo[i] - p0[i]) * ood
            t2 = (box.hi[i] - p0[i]) * ood
            if t1 > t2:
                t1, t2 = t2, t1
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


# --------------------
# Main Env
# --------------------
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
        self.tau = 0.75  # confidence parameter for obs_count

        self.reset()
    
    def reset(self):
        self.step_count = 0
        self._next_tid = 0
        self.tasks.clear()

        self.log_odds.fill(0.0)
        self.obs_count.fill(0)

        self._spawn_obstacles(n_obs=5)
        self._spawn_robots()
        for r in self.robots:
            r.traj.clear()
            r.push_traj()

        obs = self._make_obs()
        return obs
    
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
        N_uav, N_ugv = 3, 3
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

        info = {
            "step": self.step_count,
            "explore_score": float(explore_score),
            "comm_score": float(comm_score),
            "num_tasks": int(len(self.tasks)),                    
        }
        return obs, info
    
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
    # Snapshot Builder
    # -------------------------
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
            plt.show()

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

