from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
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
    z: int
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
    eta: float               # 惩罚因子 影响q-step 二次项 u-step尺度和residual计算
                             # C-ADMM验证实验可选用0.5/1.0/1.2 后续加入RL后作为可学习参数
    alpha: float             # 过松弛系数 在FeatureFlags.enable_over_relax=True时生效
                             # 设置为1.0为关闭，设置为1.5-1.8为加速
    eps_pri: float           # 原始残差阈值  1e-3
    eps_dual: float          # 对偶残差阈值  1e-3
    k_max: int               # 最大迭代轮数
                             # 固定场景设置为50-200
    ttl_hops: int            # 多跳 TTL (TTL_X)
    t_fresh: int             # 信息新鲜度窗口
    # 自适应 eta 参数 mu, tau_incr, tau_decr 
    # FeatureFlags.enable_residual_balancing=True 时使用
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

    # new added by JiXX at 20260225
    staleness_strict: bool = False  # stale判定开关
    eps_rel: float = 0.0
    assembled_eps: float = 1e-8
    root_id_default: int = 0

    # Stage0 / reachability / assembled / coupled residual
    ref_rate_mode: str = "capacity"
    ref_rate_ratio: float = 1.0
    coupled_residual_scale: float = 1.0
    diag_group_scale: float = 1.0

    # QoS normalization
    budget_cache_ema: float = 1.0

    # Beta for qstep added by JiXX at 20260227
    y_hat_beta: float = 0.1
    sigma_beta: float = 0.05


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
# updated by JiXX at 20260202
@dataclass
class FeatureFlags:
    # blocks
    # 决定在make_registry中是否包含指定的block 从而改变整体维度
    enable_pos: bool = True
    enable_cov: bool = True
    enable_f_hat: bool = True
    enable_B_hat: bool = True
    enable_y_hat: bool = True
    enable_sigma: bool = True
    enable_r_hat: bool = True

    # projections per block
    # 若 False 该 block 在 z-step 不投影
    enable_proj_pos: bool = True
    enable_proj_f_hat: bool = True
    enable_proj_B_hat: bool = True
    enable_proj_y_hat: bool = True
    enable_proj_sigma: bool = True
    enable_proj_r_hat: bool = True

    # numeric
    enable_over_relax: bool = True          # 启用over-relax z-step输入
    enable_residual_balancing: bool = True  # 启用eta自适应
    enable_async_updates: bool = False      # 启用异步mask
    enable_link_freeze: bool = True         # 启用window freeze
    enable_ttl_filter: bool = False         # 启用stale边删除
    enable_staleness_engine: bool = True    # 启用last_seen/is_stale/active_hint 的内部维护

    # q-step switches
    enable_qstep_admm_term: bool = False
    enable_qstep_admm_move: bool = False
    enable_qstep_admm_explore: bool = False
    enable_qstep_admm_task: bool = False
    enable_qstep_admm_qos: bool = False 
    enable_qstep_admm_repulsion: bool = False
    # enable beta
    enable_qstep_damped_y_hat: bool = False
    enable_qstep_damped_sigma: bool = False

    # q-step block-wise update
    enable_qstep_cost_move: bool = True        # 移动代价权重开关
    enable_qstep_cost_explore: bool = True     # 探索得分权重开关
    enable_qstep_cost_task: bool = True        # 任务紧急程度权重开关
    enable_qstep_cost_repulsion: bool = True   # 排斥代价权重开关
    enable_qstep_cost_qos_pos: bool = True     # qos代价权重开关
    enable_qstep_update_y_hat: bool = True
    enable_qstep_update_B_hat: bool = True
    enable_qstep_update_f_hat: bool = True
    enable_qstep_update_r_hat: bool = True
    enable_qstep_update_sigma: bool = True
    enable_rhat_include_coverage: bool = True
    # q-step 会按这些 flags 决定是否把 y_hat/sigma/B_hat/f_hat/r_hat 写回 q_i_new

    # residual 
    linear_assembly_mode: str = "consensus_equiv" # | "assembled_ops"
    use_linear_residual: bool = False             # 是否启用 LinearResidual
    use_coupled_linear_assembly: bool = False     # assembled_ops 残差是否乘 owner mask
    log_linear_groups: bool = False               # 是否附加 diag_* groups
    include_diag_groups_in_stop: bool = False     # diag 是否计入 total norm（影响 stop 与 eta balancing）
    enable_coupled_groups: bool = False           # 是否启用 StageE 的 coupled_* groups（sigma_y、flow_budget）
    include_coupled_groups_in_stop: bool = False  # coupled_* 是否计入 total norm 语义 stop对比的核心开关

    # diagnostics
    log_block_residuals: bool = True
    log_projection_violation: bool = True

    # new added by JiXX at 20260225
    reachability_root_id: int = 0  # 外部赋值 代表目标机器人id

    enable_stage0_init: bool = True
    enable_assembled_ops: bool = False          # z-step 使用 owner-mask assembled average
    # owner mask 与 delta 参与范围
    assembled_owner_mode: str = "all"  # | "pos_only" | "edge_by_src" | "edge_by_incident" | "root_only"
    assembled_avg_active_only: bool = False
    assembled_u_update_active_only: bool = False

    # 预算闭环统计
    budget_update_source: str = "z"
    enable_budget_cache_update: bool = False
    enable_budget_soft_violation: bool = False

    # 只影响 attach_stagec_diagnostics 里是否记录归一化预算摘要，不改变求解本体
    enable_budget_normalization: bool = False

    enable_qos_aware_edges: bool = True

    enable_flow_coupled_to_budget: bool = False  # 是否在 constraints 里收紧 f_hat 上界
    enable_sigma_coupled_to_y: bool = True       # sigma 的下界由 y_hat 诱导的开关

    async_update_u_all: bool = True

    proj_method: str = "dykstra"
    proj_iters: int = 80
    proj_tol: float = 1e-6

    active_hint_mode: str = "incident"  # | "reachability" | "reachability_with_memory"

    # reach mode
    reachability_root_mode: str = "base0"  # | "given" | "highest_degree"
    reachability_requires_fresh: bool = True
    reachability_fallback_to_incident: bool = True

    # async mode
    async_mode: str = "all"  # | "ttl_freshness" | "round_robin" | "random_k"
    async_k: int = 0  # 1
    async_ratio: float = 1.0
    
    coupled_primal_source: str = "z"  # | "q_mean"

    coord_dim: int = 2

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
    task_pos: np.ndarray       # (T, 2) -> (T, 3)  float/int grid coords
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
    robot_pos: np.ndarray               # (N, 2) -> (N, 3)
    candidate_moves: List[np.ndarray]   # len N, each (Mi, 2) -> (Mi, 3)
    coverage: float                     
    frontier_entropy: np.ndarray        # (H, W) -> (Z, H, W)
    repulsion_grad: np.ndarray          # (N, 2) -> (N, 3)

    link: LinkSnapshot
    task: TaskSnapshot

    # reg: BlockRegistry
    params: "CadmmParams"
    flags: FeatureFlags
    window: "CommWindowState"

    coord_dim: int
    
# Cadmm inner solution
@dataclass
class InnerSolution:
    next_pos: np.ndarray        # (N, 2) -> (N, 3) decoded from z_pos or q_pos
    z: np.ndarray               # (D,)
    q: np.ndarray               # (N,D)
    u: np.ndarray               # (N,D)


@dataclass
class CommWindowState:
    W: int
    start_step: int
    omega_seed: int
    frozen_link: LinkSnapshot
    budget_cache: Dict[Tuple[int, int], float] = field(default_factory=dict)
    # Outer set
    # stage0_inited_step/ref_rate/ref_total/ref_rate_summary (stage0_init)
    # last_seen_step/is_stale/active_hint (staleness/window.apply_ttl_filter)
    # parent/dist_to_root/last_success_hops/last_success_step (reachability_with_memory)
    # budget_violation/budget_violation_vec (qos_metrics.update_budget_cache)

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

