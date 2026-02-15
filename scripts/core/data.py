from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import numpy as np



@dataclass
class RLAction:
    role_ratio: np.ndarray  # [rho_main, rho_relay, rho_free]
    role_assign: Dict  # 例如 {0: "main", 1: "relay", 2: "free", ...}
    # 可选：任务偏好 / 优先级等

@dataclass
class EnvState:
    global_map: np.ndarray
    explored_map: np.ndarray
    base_pos: list
    robots: list
    tasks: list
    t_n: int
    window_horizon: int
    current_step: int

@dataclass
class Task:
    id: int
    x: int
    y: int
    start_step: int
    ddl_step: int
    status: str = "pending"  # pending | observed | completed | failed
    consec_seen: int = 0
    complete_step = None
    priority: float = 1.0
    isOnline: bool = False
    initRid: Any = None
    ref_path: Any = None
    chain_neighbors: Any = None


@dataclass
class CadmmParams:
    eta: float               # 惩罚因子
    alpha: float             # 过松弛系数
    eps_pri: float           # 原始残差阈值
    eps_dual: float          # 对偶残差阈值
    k_max: int               # 最大迭代轮数
    ttl_hops: int            # 多跳 TTL (TTL_X)
    t_fresh: int             # 信息新鲜度窗口
    # 自适应 eta 参数 mu, tau_incr, tau_decr
    mu: float
    tau_incr: float
    tau_decr: float
    # === cost ===
    move_w: float            # 移动代价权重
    
    cover_gamma: float       # 全局覆盖代价权重
    new_w: float             # 发现新cell得分权重
    fe_w: float              # 边界距离代价权重

    role_task_weight: float  # 角色任务权重
    role_relay_weight: float # 角色中继权重
    role_main_weight: float  # 角色主轴权重

    qos_exec_weight: float
    qos_relay_weight: float
    qos_default_weight: float

    qos_break_w: float
    qos_improve_w: float
    qos_degrade_w: float

    urgency_w: float         # 任务紧急程度权重

    rep_w: float             # 势能权重
    rep_sigma: float         # 机器人节点距离势能


@dataclass
class LinkState:
    tx: Any
    rx: Any
    rssi: int
    capacity: float
    delay: float
    plr: float

@dataclass
class MainFlow:
    start: Any
    relays: List
    flow: float

@dataclass
class RelayNode:
    prev: Any
    next: Any
    value: Any

# ===== cadmm block =====
# old version
# TODO: 删除并迁移外部代码到更新后的 BlockRegistry
@dataclass
class BlockSpec:
    name: str        # pos, cov, flow, qos, energy, task
    start: int       # start index in global z
    end: int         # end index in global z
    shape: tuple     # original shape of this block

# TODO: 删除并迁移外部代码到更新后的 LocalQ
@dataclass
class LocalQ:
    pos: np.ndarray
    flow=None
    qos=None
    energy=None
    tasks=None

    def copy(self) -> "LocalQ":
        return LocalQ(
            pos=self.pos.copy(),
            # flow=None if self.flow is None else self.flow.copy(),
            # qos=None if self.qos is None else self.qos.copy(),
            # energy=None if self.energy is None else self.energy.copy(),
            # tasks=None if self.tasks is None else self.tasks.copy()
        )

# TODO: 删除并迁移外部代码到更新后的 LocalU
@dataclass
class LocalU:
    pos: np.ndarray
    flow=None
    qos=None
    energy=None
    tasks=None

    def copy(self) -> "LocalU":
        return LocalU(
            pos=self.pos.copy(),
            # flow=None if self.flow is None else self.flow.copy(),
            # qos=None if self.qos is None else self.qos.copy(),
            # energy=None if self.energy is None else self.energy.copy(),
            # tasks=None if self.tasks is None else self.tasks.copy()
        )
    
# updated by JiXX at 20260202
@dataclass
class FeatureFlags:
    # blocks
    enable_pos: bool = True
    enable_cov: bool = True
    enable_f_hat: bool = True
    enable_B_hat: bool = True
    enable_y_hat: bool = True
    enable_sigma: bool = True
    enable_r_hat: bool = True

    # projections per block
    enable_proj_pos: bool = True
    enable_proj_f_hat: bool = True
    enable_proj_B_hat: bool = True
    enable_proj_y_hat: bool = True
    enable_proj_sigma: bool = True
    enable_proj_r_hat: bool = True

    # numeric
    enable_over_relax: bool = True
    enable_residual_balancing: bool = True
    enable_async_updates: bool = False
    enable_link_freeze: bool = True
    enable_ttl_filter: bool = False

    # q-step switches
    enable_qstep_admm_term: bool = True
    enable_qstep_admm_move: bool = False
    enable_qstep_admm_explore: bool = False
    enable_qstep_admm_task: bool = False
    enable_qstep_admm_qos: bool = False 
    enable_qstep_admm_repulsion: bool = False

    # diagnostics
    log_block_residuals: bool = True
    log_projection_violation: bool = True

@dataclass(frozen=True)
class BlockDef:
    name: str
    shape: tuple
    dtype: str = "float32"

# Constraints block for projections
@dataclass
class BoxC:
    # box constraints
    lo: np.ndarray
    hi: np.ndarray

@dataclass
class SimplexC:
    # 概率单纯形
    dim: int
    sum_to: float

@dataclass
class WeightedSimplexC:
    # 加权单纯形
    w: np.ndarray
    sum_value: float

@dataclass
class HalfspaceC:
    # 线性半空间约束
    a: np.ndarray
    b: float

@dataclass
class AffineC:
    # 仿射空间约束
    A: np.ndarray
    b: np.ndarray

@dataclass
class PolytopeC:
    # 多面体集合约束
    G: np.ndarray
    h: np.ndarray
    lo: Optional[np.ndarray] = None
    hi: Optional[np.ndarray] = None


# Inner problem snapshots
@dataclass
class LinkSnapshot:
    robot_ids: List[int]
    edges: np.ndarray      # (E, 2), int 32, each row (i, j) with i < j or directed, robot indices
    signal: np.ndarray     # (E, )
    # QoS metrics
    capacity: np.ndarray   # (E, )
    delay: np.ndarray      # (E, )
    plr: np.ndarray        # (E, )
    is_stale: np.ndarray   # (E, ), bool for ttl filter

@dataclass
class TaskSnapshot:
    task_pos: np.ndarray       # (T, 2)  float/int grid coords
    deadline: np.ndarray       # (T,)
    priority: np.ndarray       # (T,)
    cluster_id: np.ndarray     # (T,) int, in [0,G-1]

@dataclass
class CadmmProblem:
    # static sizes
    N: int                    # number of robots
    E: int                    # links in snapshot
    G: int                    # groups for y_hat
    T: int                    # tasks

    # inputs
    robot_pos: np.ndarray               # (N, 2)
    candidate_moves: List[np.ndarray]   # len N, each (Mi, 2)
    coverage: float                     
    frontier_entropy: np.ndarray        # (H, W)
    repulsion_grad: np.ndarray          # (N, 2)

    link: LinkSnapshot
    task: TaskSnapshot

    # reg: BlockRegistry
    params: "CadmmParams"
    flags: FeatureFlags
    window: "CommWindowState"
    
# Cadmm inner solution
@dataclass
class InnerSolution:
    next_pos: np.ndarray        # (N,2) decoded from z_pos or q_pos
    z: np.ndarray               # (D,)
    q: np.ndarray               # (N,D)
    u: np.ndarray               # (N,D)


@dataclass
class CommWindowState:
    W: int
    start_step: int
    omega_seed: int
    frozen_link: LinkSnapshot

# Log and Warm-start
@dataclass
class CadmmWarmStart:
    z: np.ndarray             # (D, )
    u: np.ndarray             # (N, D)
    q: np.ndarray             # (N, D)

@dataclass
class CadmmDiagnostics:
    iters: int
    r_norm: Dict[str, float]       # block->r
    s_norm: Dict[str, float]       # block->s
    eta_hist: List[float]
    proj_violation: Dict[str, float]

