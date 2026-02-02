from dataclasses import dataclass
from typing import Dict, List, Any
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
class CadmmWarmStart:
    q_prev: Dict  # 每个机器人上一轮的局部变量
    z_prev: np.ndarray             # 全局变量
    u_prev: Dict  # 对偶变量

# Interface
@dataclass
class RobotPlan:
    waypoints: List   # 规划窗口内的轨迹 [(x_t, y_t), ...]
    role: str                          # "main" / "relay" / "free"
    assigned_tasks: List          # 当前窗口负责的任务 ID

@dataclass
class InnerPlan:
    robot_plans: Dict  # 每个机器人一份计划
    dual_state: CadmmWarmStart         # 用于下一轮 warm-start
    diagnostics: dict       # 残差曲线 / 目标值 / 迭代次数等


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
@dataclass
class BlockSpec:
    name: str        # pos, cov, flow, qos, energy, task
    start: int       # start index in global z
    end: int         # end index in global z
    shape: tuple     # original shape of this block

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