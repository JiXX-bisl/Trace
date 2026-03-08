from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Callable
import numpy as np
import os
import json
import time
from pathlib import Path


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

    obst_w: float
    rho: float

    urgency_w: float         # 任务紧急程度权重

    rep_w: float             # 势能权重
    rep_sigma: float         # 机器人节点距离势能

    qos_w: float

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
    sigma_beta: float = 0.0

    # theta QP projected gradient settings
    # Enable only when flags.enable_theory_mode and flags.enable_theta
    theta_pg_iters: int = 8             # projected gradient iterations
    theta_eta_y: float = 1.0            # eta_y = theta_eta_y * params.eta
    theta_eta_prior: float = 0.1        # stabilizer: eta_theta * || theta - theta_prev ||^2
    theta_step: Optional[float] = None  # None -> auto step size
    
    # new added by JiXX at 20260302
    los_max_dist: float = 15.0
    C_max: float = 30.0
    # Y average-assembly penalty scaling (for strict theory alignment):
    # eta_y = eta_y_avg_scale * params.eta
    eta_y_avg_scale: float = 1.0

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
    enable_qstep_cost_obstacles: bool = False
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

    # Theory-alignment switches (all disabled by default for backward compat)
    # enable_theory_mode:
    #   - When True, solver/residual should follow the document convention:
    #     u-step and residuals are updated against z_eq (consensus variable)
    enable_theory_mode: bool = False
    # enable_sigma_y_joint_polytope:
    #   - When True, z-step should project (y_hat, sigma) jointly onto the
    #     polytope implied by the theory coupling constraints.
    enable_sigma_y_joint_polytope: bool = False
    #   - In theory mode, y_hat should use box constraint [0,1] (not simplex).
    theory_y_hat_box_only: bool = False
     #   - Prepare for strict alignment where y_hat is stacked over time: dim(y_hat) = |G| * T_horizon.
    enable_time_stacked_y_hat: bool = False
    # Route-B: convexified discrete action weights theta
    enable_theta: bool = False
    theta_dim: int = 0
    enable_qstep_update_theta: bool = True
    # execution-layer option: whether to execute argmax(theta) as a discrete action
    theta_use_argmax_exec: bool = True
    # Strict theory alignment: Y-block uses average-assembly (mean(s_hat) - y_hat) instead of consensus (q_y - z_y)
    # Enabled only when enable_theory_mode and enable_theta are also enabled.
    enable_y_avg_assembly: bool = False

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
    G: int                    # number of groups for y_hat
    T: int                    # number of tasks

    # inputs
    robots: Any
    robot_pos: np.ndarray               # (N, 2) -> (N, 3)
    candidate_moves: List[np.ndarray]   # len N, each (Mi, 2) -> (Mi, 3)
    coverage: float                     
    frontier_entropy: np.ndarray        # (H, W) -> (Z, H, W)
    grid_gain: np.ndarray
    frontier_pts: np.ndarray
    # repulsion_grad: np.ndarray          # (N, 2) -> (N, 3)

    link: LinkSnapshot
    task: TaskSnapshot

    obstacle_lo: np.ndarray
    obstacle_hi: np.ndarray

    # reg: BlockRegistry
    params: "CadmmParams"
    flags: FeatureFlags
    window: "CommWindowState"

    is_los_fn: Callable[[np.ndarray, np.ndarray], bool]

    coord_dim: int
    # new added
    K: Optional[int] = None
    T_horizon: int = 1

    def __post_init__(self) -> None:
        # Backward compat: if caller only provides legacy T, mirror to K.
        if self.K is None: self.K = int(self.T)
        # Keep legacy T coherent for old code paths that still read problem T
        if (int(self.T) <= 0) and (self.K is not None): self.T = int(self.K)
        # Sanitize time horizon
        try:
            self.T_horizon = int(self.T_horizon)
        except Exception:
            self.T_horizon = 1
        if self.T_horizon < 1: self.T_horizon = 1
    
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


# 待添加量
# rep_mode = getattr(params, "rep_mode", "quad")
# rep_d0   = float(getattr(params, "rep_d0", 3.0))
# rep_sig  = float(getattr(params, "rep_sigma", 1.0))
# sigma_rep = {"mode": rep_mode, "d0": rep_d0, "sigma": rep_sig}

# rep_cost = cost_repulsion(cand, current_pos, problem.robot_pos, sigma_rep, rep_w)

# =============================================================================
# Cadmm logging (rollout-level aggregation)
# =============================================================================

def _to_py(v: Any):
    """Best-effort conversion to JSON-serializable python types."""
    if v is None:
        return None
    if isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        # store small arrays inline; large arrays should go to .npz
        if v.size <= 64:
            return v.tolist()
        return {"__ndarray__": True, "shape": list(v.shape), "dtype": str(v.dtype)}
    if isinstance(v, dict):
        return {str(k): _to_py(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_py(x) for x in v]
    # dataclass?
    try:
        from dataclasses import asdict, is_dataclass
        if is_dataclass(v):
            return _to_py(asdict(v))
    except Exception:
        pass
    return str(v)

@dataclass
class CadmmLogConfig:
    """Logging configuration.

    level:
      - 'lite': store per-iter scalars + selected z-block trajectories.
      - 'full': additionally store z/q/u (optionally downcasted) every `stride`.
    """
    level: str = "lite"             # 'lite' | 'full'
    stride: int = 1                 # store every k%stride==0
    store_blocks: Tuple[str, ...] = ("pos", "f_hat", "B_hat", "y_hat", "sigma", "r_hat")
    store_full_state: bool = False  # store z/q/u arrays (can be huge)
    state_dtype: str = "float32"    # 'float32' | 'float16'
    save_compressed: bool = True

@dataclass
class CadmmRunLog:
    """One inner C-ADMM solve (one snapshot / one env step)."""
    run_id: int
    rollout_step: int
    mode: str                        # 'theory' | 'legacy'
    timestamp: float

    # Snapshot meta (small; arrays go to npz if needed)
    N: int
    E: int
    G: int
    K: int
    T_horizon: int
    coord_dim: int
    block_order: List[str] = field(default_factory=list)
    block_slices: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    flags: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    window: Dict[str, Any] = field(default_factory=dict)

    # Initial state (optional)
    z0: Optional[np.ndarray] = None
    q0: Optional[np.ndarray] = None
    u0: Optional[np.ndarray] = None

    # Per-iter scalars
    it_k: List[int] = field(default_factory=list)
    it_eta: List[float] = field(default_factory=list)
    it_r_stop: List[float] = field(default_factory=list)
    it_s_stop: List[float] = field(default_factory=list)
    it_eps_pri: List[float] = field(default_factory=list)
    it_eps_dual: List[float] = field(default_factory=list)

    # Per-iter per-block vectors (aligned to block_order)
    it_r_by_block: List[np.ndarray] = field(default_factory=list)
    it_s_by_block: List[np.ndarray] = field(default_factory=list)
    it_proj_vio: List[np.ndarray] = field(default_factory=list)

    # Async masks
    it_active_mask: List[np.ndarray] = field(default_factory=list)
    it_u_mask: List[np.ndarray] = field(default_factory=list)

    # Coupled pass markers / summaries
    it_coupled: List[Dict[str, Any]] = field(default_factory=list)

    # Selected z-block trajectories (aligned to store_blocks)
    it_z_blocks: Dict[str, List[np.ndarray]] = field(default_factory=dict)

    # Optional full state trajectories
    it_z: List[np.ndarray] = field(default_factory=list)
    it_q: List[np.ndarray] = field(default_factory=list)
    it_u: List[np.ndarray] = field(default_factory=list)

    # Final outputs (stored once)
    final_next_pos: Optional[np.ndarray] = None
    final_z: Optional[np.ndarray] = None
    final_q: Optional[np.ndarray] = None
    final_u: Optional[np.ndarray] = None
    final_diag: Dict[str, Any] = field(default_factory=dict)

    def append_iter(
        self,
        *,
        k: int,
        eta: float,
        r_stop: float,
        s_stop: float,
        eps_pri: float,
        eps_dual: float,
        block_order: List[str],
        r_by_block: Dict[str, float],
        s_by_block: Dict[str, float],
        proj_violation: Dict[str, float],
        active_mask: Optional[np.ndarray],
        u_mask: Optional[np.ndarray],
        coupled_summary: Optional[Dict[str, Any]],
        z: Optional[np.ndarray],
        q: Optional[np.ndarray],
        u: Optional[np.ndarray],
        store_blocks: Tuple[str, ...],
        store_full_state: bool,
        state_dtype: str,
    ) -> None:
        self.it_k.append(int(k))
        self.it_eta.append(float(eta))
        self.it_r_stop.append(float(r_stop))
        self.it_s_stop.append(float(s_stop))
        self.it_eps_pri.append(float(eps_pri))
        self.it_eps_dual.append(float(eps_dual))

        # vectors aligned to block order
        r_vec = np.zeros((len(block_order),), dtype=np.float32)
        s_vec = np.zeros((len(block_order),), dtype=np.float32)
        p_vec = np.zeros((len(block_order),), dtype=np.float32)
        for ii, name in enumerate(block_order):
            r_vec[ii] = float(r_by_block.get(name, 0.0) or 0.0)
            s_vec[ii] = float(s_by_block.get(name, 0.0) or 0.0)
            p_vec[ii] = float(proj_violation.get(name, 0.0) or 0.0)
        self.it_r_by_block.append(r_vec)
        self.it_s_by_block.append(s_vec)
        self.it_proj_vio.append(p_vec)

        if active_mask is not None:
            self.it_active_mask.append(np.asarray(active_mask, dtype=np.uint8).copy())
        if u_mask is not None:
            self.it_u_mask.append(np.asarray(u_mask, dtype=np.uint8).copy())

        self.it_coupled.append(dict(coupled_summary or {}))

        # Selected z-blocks
        if (z is not None) and (store_blocks is not None):
            for name in store_blocks:
                if name not in self.block_slices:
                    continue
                sl0, sl1 = self.block_slices[name]
                blk = np.asarray(z[sl0:sl1], dtype=np.float32).copy()
                self.it_z_blocks.setdefault(name, []).append(blk)

        # Full state trajectories (optional)
        if store_full_state and (z is not None) and (q is not None) and (u is not None):
            dt = np.float16 if str(state_dtype).lower() == "float16" else np.float32
            self.it_z.append(np.asarray(z, dtype=dt).copy())
            self.it_q.append(np.asarray(q, dtype=dt).copy())
            self.it_u.append(np.asarray(u, dtype=dt).copy())

    def finalize(self, *, sol: Any, diag: Any) -> None:
        try:
            self.final_next_pos = np.asarray(getattr(sol, "next_pos", None), dtype=np.float32) if getattr(sol, "next_pos", None) is not None else None
            self.final_z = np.asarray(getattr(sol, "z", None), dtype=np.float32) if getattr(sol, "z", None) is not None else None
            self.final_q = np.asarray(getattr(sol, "q", None), dtype=np.float32) if getattr(sol, "q", None) is not None else None
            self.final_u = np.asarray(getattr(sol, "u", None), dtype=np.float32) if getattr(sol, "u", None) is not None else None
        except Exception:
            pass
        # diagnostics (jsonable)
        try:
            self.final_diag = _to_py(diag)
        except Exception:
            self.final_diag = {}

    def save(self, out_dir: str, *, cfg: CadmmLogConfig) -> None:
        outp = Path(out_dir)
        outp.mkdir(parents=True, exist_ok=True)

        meta = {
            "run_id": self.run_id,
            "rollout_step": self.rollout_step,
            "mode": self.mode,
            "timestamp": self.timestamp,
            "sizes": {"N": self.N, "E": self.E, "G": self.G, "K": self.K, "T_horizon": self.T_horizon, "coord_dim": self.coord_dim},
            "block_order": list(self.block_order),
            "block_slices": {k: [int(v[0]), int(v[1])] for k, v in self.block_slices.items()},
            "flags": self.flags,
            "params": self.params,
            "window": self.window,
        }
        (outp / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

        # Pack arrays
        arrays: Dict[str, Any] = {}
        arrays["k"] = np.asarray(self.it_k, dtype=np.int32)
        arrays["eta"] = np.asarray(self.it_eta, dtype=np.float32)
        arrays["r_stop"] = np.asarray(self.it_r_stop, dtype=np.float32)
        arrays["s_stop"] = np.asarray(self.it_s_stop, dtype=np.float32)
        arrays["eps_pri"] = np.asarray(self.it_eps_pri, dtype=np.float32)
        arrays["eps_dual"] = np.asarray(self.it_eps_dual, dtype=np.float32)

        if self.it_r_by_block:
            arrays["r_by_block"] = np.stack(self.it_r_by_block, axis=0)
        if self.it_s_by_block:
            arrays["s_by_block"] = np.stack(self.it_s_by_block, axis=0)
        if self.it_proj_vio:
            arrays["proj_vio"] = np.stack(self.it_proj_vio, axis=0)

        # masks
        if self.it_active_mask:
            arrays["active_mask"] = np.stack(self.it_active_mask, axis=0)
        if self.it_u_mask:
            arrays["u_mask"] = np.stack(self.it_u_mask, axis=0)

        # coupled summaries as jsonl (more robust than np object arrays)
        with (outp / "coupled.jsonl").open("w", encoding="utf-8") as f:
            for row in self.it_coupled:
                f.write(json.dumps(_to_py(row), ensure_ascii=False) + "\n")

        # selected blocks
        for name, lst in self.it_z_blocks.items():
            if lst:
                arrays[f"zblk_{name}"] = np.stack(lst, axis=0)

        # init/final states (optional)
        if self.z0 is not None: arrays["z0"] = np.asarray(self.z0, dtype=np.float32)
        if self.q0 is not None: arrays["q0"] = np.asarray(self.q0, dtype=np.float32)
        if self.u0 is not None: arrays["u0"] = np.asarray(self.u0, dtype=np.float32)
        if self.final_next_pos is not None: arrays["final_next_pos"] = np.asarray(self.final_next_pos, dtype=np.float32)
        if self.final_z is not None: arrays["final_z"] = np.asarray(self.final_z, dtype=np.float32)

        if cfg.level == "full":
            # store full trajectories if present
            if self.it_z: arrays["z_traj"] = np.stack(self.it_z, axis=0)
            if self.it_q: arrays["q_traj"] = np.stack(self.it_q, axis=0)
            if self.it_u: arrays["u_traj"] = np.stack(self.it_u, axis=0)

        npz_path = outp / "arrays.npz"
        if cfg.save_compressed:
            np.savez_compressed(npz_path, **arrays)
        else:
            np.savez(npz_path, **arrays)

        # final diag
        (outp / "final_diag.json").write_text(json.dumps(self.final_diag, indent=2, ensure_ascii=False))

@dataclass
class CadmmLog:
    """Rollout-level logger for inner C-ADMM.

    Typical usage:
        log = CadmmLog()
        problem.log = log
        sol, ws, diag = solve_inner_cadmm(problem)
        ...
        log.save("out_dir")
    """
    config: CadmmLogConfig = field(default_factory=CadmmLogConfig)
    runs: List[CadmmRunLog] = field(default_factory=list)

    def start_run(self, *, problem: Any, reg: Any, z0: np.ndarray, q0: np.ndarray, u0: np.ndarray) -> CadmmRunLog:
        rid = len(self.runs)
        mode = "theory" if bool(getattr(getattr(problem, "flags", None), "enable_theory_mode", False)) else "legacy"
        step = int(getattr(problem, "rollout_step", getattr(problem, "step", -1)))
        if step < 0:
            # fallback: window start step
            try:
                step = int(getattr(getattr(problem, "window", None), "start_step", -1))
            except Exception:
                step = -1

        run = CadmmRunLog(
            run_id=rid,
            rollout_step=step,
            mode=mode,
            timestamp=float(time.time()),
            N=int(getattr(problem, "N", 0)),
            E=int(getattr(problem, "E", 0)),
            G=int(getattr(problem, "G", 0)),
            K=int(getattr(problem, "K", getattr(problem, "T", 0)) or 0),
            T_horizon=int(getattr(problem, "T_horizon", 1) or 1),
            coord_dim=int(getattr(problem, "coord_dim", 0) or 0),
        )
        # registry meta
        try:
            run.block_order = list(reg.names())
            for name in run.block_order:
                sl = reg.sl(name)
                run.block_slices[name] = (int(sl.start), int(sl.stop))
        except Exception:
            pass

        # flags/params/window (jsonable subset)
        try:
            run.flags = _to_py(getattr(problem, "flags", {}))
        except Exception:
            run.flags = {}
        try:
            run.params = _to_py(getattr(problem, "params", {}))
        except Exception:
            run.params = {}
        try:
            w = getattr(problem, "window", None)
            run.window = _to_py({
                "W": getattr(w, "W", None),
                "start_step": getattr(w, "start_step", None),
                "omega_seed": getattr(w, "omega_seed", None),
            })
        except Exception:
            run.window = {}

        run.z0 = np.asarray(z0, dtype=np.float32).copy()
        run.q0 = np.asarray(q0, dtype=np.float32).copy()
        run.u0 = np.asarray(u0, dtype=np.float32).copy()

        self.runs.append(run)
        return run

    def save(self, out_dir: str) -> None:
        outp = Path(out_dir)
        outp.mkdir(parents=True, exist_ok=True)
        idx = {
            "version": 1,
            "created": float(time.time()),
            "config": _to_py(self.config),
            "num_runs": len(self.runs),
        }
        (outp / "index.json").write_text(json.dumps(idx, indent=2, ensure_ascii=False))
        runs_dir = outp / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        for run in self.runs:
            run_dir = runs_dir / f"run_{run.run_id:04d}"
            run.save(str(run_dir), cfg=self.config)

