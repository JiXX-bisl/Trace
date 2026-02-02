"""
cadmm_solver.py
Created by JiXX at 20251203
We define the necessary interface and inner optimiazation process of inner CADMM and outer interaction.
- input: 
    * current env (map + robot + tasks)
    * outer RL actions
    * cost of communication
    * C-ADMM hyper parameters
    * last C-ADMM state
- main loop:
    * actions
    * global consensus
    * dual update
    * residual calc
- output:
    * a short term trajectory (moving to which cell at next step)
    * dual state
- interface:
    * selected cell
"""
from dataclasses import dataclass
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import defaultdict

from scripts.core.data import CadmmParams, CadmmWarmStart, EnvState, InnerPlan, RLAction
from scripts.environment.communication import LinkState
from scripts.environment.robot import UAV, UGV, enumerate_candidate_moves
from scripts.environment.grid_map import estimate_new_unknown_cells, compute_frontier_entropy, predict_coverage_after_moves
from scripts.utils.helper import compute_current_coverage_ratio, entropy_potential_at_pos, update_repulsion_grad


def inner_cadmm(
    env_state: EnvState,             
    rl_action: RLAction,
    link_state: List[LinkState],
    cadmm_params: CadmmParams, 
    warm_start=None
):
    """
    TRACE 内层CADMM优化，优化全局平均策略，输出每个机器人的预期轨迹和通信资源分配
    TRACE Inner-CADMM, get global average strategy, return trajectory and communication resource allocation strategy
    1 init q, z, u
    2 C-ADMM iteration
    3 get robot plan
    4 return inner plan
    """
    # Step 1 initialize q, z, u
    q, z, u, aux = _init_cadmm_state(env_state, rl_action)
    q_current = q.copy()
    # Step 2 Main optimize loop
    r_hist, s_hist = [], []
    for k in range(cadmm_params.k_max):
        # update q for each robot
        aux["q_prev"] = {idx: q_current[idx].copy() for idx in aux["robot_ids"]}
        update_repulsion_grad(aux, q_current)
        q_new = _local_update_all(q_current, z, u, env_state, rl_action, link_state, cadmm_params, aux)
        # global update z
        z_new = _global_update_z(q_new, z, env_state, cadmm_params, aux)

        # dual update and residual calculate
        u, r_val, s_val, converged  = _dual_update_residual(q_new, z, z_new, u, cadmm_params, aux)
        r_hist.append(r_val)
        s_hist.append(s_val)

        # ending judegement
        if converged: break

        z = z_new
        q_current = q_new

    # Step 3 decoding robot trajectory from robot plan
    robot_plans = _decode_robot_plan(q_current, env_state, rl_action, aux)

    dual_state = CadmmWarmStart(q_prev=q_current, z_prev=z_new, u_prev=u)
    diagnostics = {
        "r_hist": np.array(r_hist), 
        "s_hist": np.array(s_hist), 
        "num_iter": len(r_hist)
    }

    return InnerPlan(robot_plans=robot_plans, dual_state=dual_state, diagnostics=diagnostics)

def _init_cadmm_state(env_state: EnvState, rl_action=None, warm_start=None):
    """
    Initialize ADMM state, including: local q, global z, dual u and helper aux
    - q: q_i denotes next step pos for each robot
    - z: cat of positions and global coverage ratio
    - u: 0
    TODO: add link state, QoS state to z
    """
    robot_ids = [robot.id for robot in env_state.robots]
    num_robots = len(robot_ids)

    # 1) get action candidate for each robot
    candidate_moves: Dict[int, List[tuple[int, int]]] = {}  # id: [x, y]
    for rid in robot_ids:
        candidate_moves[rid] = enumerate_candidate_moves(env_state, rid)

    # 2) initialize q and u for each robot
    q: Dict[int, np.ndarray] = {}
    u: Dict[int, np.ndarray] = {}

    # initialize global z
    if warm_start is not None:
        for rid in robot_ids:
            if rid in warm_start.q_prev:
                q[rid] = warm_start.q_prev[rid].copy()
            else:
                x, y = env_state.robots[rid].pos
                q[rid] = np.array([float(x), float(y)], dtype=float)
            if rid in warm_start.u_prev:
                u[rid] = warm_start.u_prev[rid].copy()
            else:
                u[rid] = np.zeros(2, dtype=float)
        expected_dim = 2 * num_robots + 1
        if warm_start.z_prev.shape[0] == expected_dim:
            z = warm_start.z_prev.copy()
        else:
            z_pos_coords: List[float] = []
            for rid in robot_ids:
                x, y = env_state.robots[rid].pos
                z_pos_coords.extend([float(x), float(y)])
            z_pos = np.array(z_pos_coords, dtype=float)
            z_cov = compute_current_coverage_ratio(env_state)
            z = np.concatenate([z_pos, np.array([z_cov], dtype=float)], axis=0)
    else:
        # ---- cold start ----
        z_pos_coords: List[float] = []

        for rid in robot_ids:
            x, y = env_state.robots[rid].pos
            q[rid] = np.array([float(x), float(y)], dtype=float)
            u[rid] = np.zeros(2, dtype=float)
            z_pos_coords.extend([float(x), float(y)])

        z_pos = np.asarray(z_pos_coords, dtype=float)
        z_cov = compute_current_coverage_ratio(env_state)
        z = np.concatenate([z_pos, np.array([z_cov], dtype=float)], axis=0)
    
    # # 3) aux
    frontier_entropy = compute_frontier_entropy(env_state)
    # print(frontier_entropy)
    aux: Dict[str, Any] = {
        "robot_ids": robot_ids,
        "num_robots": num_robots,
        "candidate_moves": candidate_moves,
        "rid_to_index": {rid: idx for idx, rid in enumerate(robot_ids)},
        "z_dim_pos": 2 * num_robots,
        "z_dim_cov": 1, 
        "frontier_entropy": frontier_entropy, 
        "q_prev": {idx: q[idx].copy() for idx in robot_ids}, 
        "repulsion_grad": {idx: np.zeros(2, dtype=float) for idx in robot_ids}, 
        "repulsion_sigma": 2.0,
        "repulsion_weight": 0.5
        # TODO(full-TRACE-zblocks):
        #   在完整版 TRACE 中，z 不仅包含位置和覆盖率，还需要包含：
        #   - z_flow: 链路流量 / 带宽分配（边/节点层面的连续变量）
        #   - z_qos: 主轴链路 QoS / TTL 约束相关的全局状态
        #   - z_energy: 能耗 / 电量相关状态
        #   - z_task: 任务执行量 / 服务量相关状态
        #   建议在这里维护一个统一的 block 索引表，方便 z-步和 u-步按 block 操作：
        #   例如：
        #   "z_blocks": {
        #       "pos": (0, 2*num_robots),
        #       "cov": (2*num_robots, 2*num_robots+1),
        #       "flow": (start_flow, end_flow),
        #       "qos": (start_qos, end_qos),
        #       "energy": (start_energy, end_energy),
        #       "task": (start_task, end_task),
        #   }
        #   当前最小版本只维护 pos / cov，后续扩展时在这里统一加索引。
        # "z_blocks": {...}
    }

    # TODO(full-TRACE-q-structure):
    #   当前 q_i 只包含位置 [x_i_next, y_i_next] ∈ R^2。
    #   在完整 TRACE 中，q_i 需要扩展为：
    #     q_i = [position, local_flow, local_task, local_energy, ...]
    #   这些局部变量对应 z_flow/z_task/z_energy 等全局 block。
    #   这里的初始化要同步扩展这些维度，并在 warm_start 中恢复。
    return q, z, u, aux

def _local_update_all(q, z, u, env_state, rl_action, link_state, cadmm_params, aux):
    """
    Get local q update for each robot i parallely
        q_i^{k+1}=argmin_{q_i \in Q_i}r_i(q_i) + (\eta/2)||q_i-z_i^k+u_i^k||^2
    Q_i: candidate actions   
    """
    new_q: Dict[int, np.ndarray] = {}
    for rid in aux["robot_ids"]:
        new_q[rid] = _solve_local(
            rid=rid,
            z=z,
            u_i=u[rid],
            env_state=env_state,
            rl_action=rl_action,
            link_state=link_state,
            cadmm_params=cadmm_params,
            aux=aux
        )
    return new_q

def _global_update_z(q, z_old, env_state, cadmm_params, aux) -> np.ndarray:
    """
    根据所有局部 q_i 更新全局变量 z。
    这里采用与自由任务(COI)和主轴任务(探索)相匹配的机器人位置与cover得分作为全局内容
    - 位置 block:
        z_pos^{k+1} = cat({q_i^{k+1}}_i \in N)
    - 覆盖率 block:
        全局覆盖率

    TODO(full-TRACE-gz-core):
    - 在完整 TRACE 中，z-步需要显式求解：
        z^{k+1} = argmin_z g(z) + (η/2)||A q^{k+1} + B z + u^k||^2
      其中 g(z) 包含：
        * QoS 相关的全局约束（例如：主轴链路的带宽 ≥ B_min，时延 ≤ D_max）
        * 全局能耗/资源约束
        * 任务执行量守恒 / 分配约束（例如 Σ_i x_{i,l} = 1）
      建议实现方式：
        1) 按 block 拆解 z：
             z = [z_pos, z_cov_vector, z_flow, z_qos, z_energy, z_task]
        2) 对每个 block 实现一个 proximal/projection：
             z_flow = Proj_{FlowFeasible}(z_flow_tilde)
             z_qos  = Proj_{QoSFeasible}(z_qos_tilde)
             ...
        3) 将这些投影操作收敛为 g(z) 的近端映射，实现真正的 ADMM z-步。
    - 与异步时变图适配：
        * 引入 c^{k+1}（过松弛中心量）和图混合矩阵 W(k)：
             c^{k+1}_i = Σ_{j∈N_i(k)} w_{ij}(k) · z_j^{k+1}
          当前集中式实现可以在 c^{k+1} 上做一次“全局平均”，
          分布式实现则用邻接矩阵邻居通信。
    """
    robot_ids = aux["robot_ids"]
    num_robots = aux["num_robots"]
    z_dim_pos = aux["z_dim_pos"]

    # 1) position block
    z_pos_new_list = []
    for rid in robot_ids:
        qi = q[rid]
        z_pos_new_list.extend(qi.tolist())
    z_pos_new = np.asarray(z_pos_new_list, dtype=float)

    # 2) cover block
    z_cov_new = predict_coverage_after_moves(env_state, q)

    # 3) over-relaxation
    alpha = cadmm_params.alpha
    # TODO(full-TRACE-overrelaxation):
    #   当前仅对位置 block 做简单的过松弛：
    #       z_pos_relaxed = α z_pos_new + (1-α) z_pos_old
    #   完整版需要：
    #     - 定义 c^{k+1} = α z^{k+1} + (1-α) z^k 并在 q-步中使用 c 而非 z；
    #     - 对异步时变图，引入混合矩阵 W(k) 对 c 做邻居聚合；
    #     - 将这些实现与理论部分的“过松弛 & 图适配”定理对应起来。
    if alpha != 1.0:
        z_pos_old = z_old[:z_dim_pos]
        z_pos_relaxed = alpha * z_pos_new + (1.0 - alpha) * z_pos_old
        z_pos_use = z_pos_relaxed
        z_cov_use = z_cov_new
    else:
        z_pos_use = z_pos_new
        z_cov_use = z_cov_new

    # 4) concatenate
    z_new = np.concatenate(
        [z_pos_use, np.array([z_cov_use], dtype=float)],
        axis=0
    )
    return z_new
    

def _dual_update_residual(q, z, z_new, u, cadmm_params, aux):
    """
    ADMM dual update and residual calculation
    - Consensus constraint on pos block
    - residual
        * original residual, r_i^{k+1} = q_i^{k+1} - z_i^{k+1} 
        * dual residual, s^{k+1} = \eta (z_pos^{k+1} - z_pos^k)
        * dual update, u_i^{k+1} = u_i^k + r_i^{k+1} 
    """
    robot_ids = aux["robot_ids"]
    num_robots = aux["num_robots"]
    z_dim_pos = aux["z_dim_pos"]

    eta = cadmm_params.eta
    eps_pri = cadmm_params.eps_pri
    eps_dual = cadmm_params.eps_dual

    # 1) get block for new and previous pos
    z_pos_new = z_new[:z_dim_pos]
    z_pos_old = z[:z_dim_pos]

    # 2) original residual r = q - z_pos_new, update dual parameter u
    r_list = []
    new_u = {}
    for rid in robot_ids:
        idx = aux["rid_to_index"][rid]

        z_i_new = z_pos_new[2 * idx: 2 * idx + 2]
        q_i  = q[rid]
        r_i = q_i - z_i_new
        r_list.extend(r_i.tolist())

        u_i_old = u[rid]
        u_i_new = u_i_old + r_i
        new_u[rid] = u_i_new
    
    r_vec = np.asarray(r_list, dtype=float)
    r_norm = float(np.linalg.norm(r_vec, ord=2))

    # 3) dual residual s = \eta (z_pos^{k+1} - z_pos^k)
    dz_pos = z_pos_new - z_pos_old
    s_vec = eta * dz_pos
    s_norm = float(np.linalg.norm(s_vec, ord=2))

    # 4) converage judgement
    converged = (r_norm <= eps_pri) and (s_norm <= eps_dual)

    # =========================
    # TODO: 需要根据完整的TRACE定义进行扩展
    # 在完整的TRACE中，需要扩展的内容包括：
    # 1) 多block残差，目前只有在位置block上定义了r和s。
    #    z在完整TRACE版本中将包含：
    #       - z_pos：位置
    #       - z_cov_vec: 区域化覆盖向量
    #       - z_flow: 链路流量
    #       - z_qos: 主轴 QoS
    #       - z_energy: 能耗
    #       - z_task: 任务执行量
    #    对每个block都需要定义
    #        r^{(block)} = A_block q - B_block z_block
    #        s^{(block)} = η B_block^T (z_block^{k+1} - z_block^k)
    #    并根据这些block进行汇总
    #        r_norm = sqrt(sum_block ||r^{(block)}||^2)
    #        s_norm = sqrt(sum_block ||s^{(block)}||^2)
    # 2) block阈值设定
    #    对于不同block，收敛精度敏感性不同，所以需要引入不同的收敛阈值作为判据
    #        eps_pri_pos, eps_pri_flow, eps_pri_qos, ...
    #    并采用:
    #        converged = all( ||r^{(block)}|| <= eps_pri_block
    #                         and ||s^{(block)}|| <= eps_dual_block )
    # 3) 异步时变图：
    #    需要维护一个B-连通时间窗口，对r_norm和s_norm做：
    #       - 窗口内最大值/均值统计
    #       - 使用统计值与阈值比较，而非单步r_norm，s_norm
    # 4) 与外层 RL 的耦合:
    #    - 当前 converged 只用于内层循环是否提前终止。
    #    - 完整 TRACE 中，可以在外层 RL 调度中使用:
    #         * 是否收敛 (converged)
    #         * r_norm, s_norm 曲线
    #      作为一种约束或诊断指标（例如：限制内层迭代次数，或在 PPO 的 reward 中加入
    #      对“内层未收敛次数”的惩罚）。
    #
    # 5) 日志记录与可视化:
    #    - 将每次 inner 迭代的 r_norm, s_norm 写入日志/存盘，
    #      便于在实验章节中给出 C-ADMM 收敛行为的可视化（如残差随迭代次数衰减曲线）。
    # =========================

    return new_u, r_norm, s_norm, converged

def _decode_robot_plan(q, env_state, rl_action, aux):
    """
    Decode results of C-ADMM to robot cmd
    - next position (nx, ny)
    """
    robot_ids = aux["robot_ids"]
    candidate_moves = aux["candidate_moves"]

    # current pos
    cur_pos = {}
    roles = {}
    for rid in robot_ids:
        x_cur, y_cur = env_state.robots[rid].pos
        cur_pos[rid] = (int(x_cur), int(y_cur))
        roles[rid] = rl_action.role_assign.get(rid, "free")

    proposals = {}
    cell_to_rids = defaultdict(list)
    plan = {}
    for rid in robot_ids:
        q_i = q[rid]
        x_target_f = float(q_i[0])
        y_target_f = float(q_i[1])
        moves = candidate_moves[rid]
        best_move = None
        best_d2 = np.infty
        for (cx, cy) in moves:
            dx = cx - x_target_f
            dy = cy - y_target_f
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_move = (cx, cy)
        if best_move is None:
            x_cur, y_cur = env_state.robots[rid].pos
            nx, ny = int(x_cur), int(y_cur)
        else:
            nx, ny = best_move
        x_cur, y_cur = env_state.robots[rid].pos
        x_cur, y_cur = int(x_cur), int(y_cur)

        dx = nx - x_cur
        dy = ny - y_cur
        role = rl_action.role_assign[rid]
        plan[rid] = {
            "rid": rid,
            "role": role,
            "next_pos": (nx, ny),          # 下一个 step 要前往的位置（栅格坐标）
            "delta": (dx, dy),             # 位移向量，方便你后面画动画 / 统计步长等
            "stay": (dx == 0 and dy == 0),
            "action_type": "move" if (dx != 0 or dy != 0) else "hold",
        }
    # ==============================
    # TODO(full-TRACE-decode-plan):
    # ==============================
    # 在完整 TRACE 部署中，这里建议进一步扩展：
    #
    # 1) 多类型动作支持：
    #    - 当前只区分 "move" / "hold"。
    #    - 完整版中可以根据 rl_action 和任务状态，扩展动作类型：
    #        * "observe"   : 在 COI 附近执行观测/停留动作（例如需要连续 2 步观测）
    #        * "relay"     : 中继节点建立/调整姿态的动作（朝向/高度/功率调整等）
    #        * "charge"    : 前往充电/补给点
    #        * "handover"  : 与其他机器人进行任务/数据交接
    #    - 在 plan[rid] 增加字段：
    #        * "task_id" / "coi_id" / "relay_link" 等，高层模块据此触发不同控制逻辑。
    #
    # 2) 与物理世界坐标系的对接：
    #    - 当前 next_pos 是栅格坐标 (i, j)。
    #    - 在真实机器人/仿真（Gazebo、ROS2）中需要转换为世界坐标：
    #        world_x = origin_x + (i + 0.5) * resolution
    #        world_y = origin_y + (j + 0.5) * resolution
    #    - 建议在 env_state 中维护 map 的 origin 和 resolution，
    #      在 decode 阶段直接给出 world_pose，便于封装成 ROS 消息：
    #        geometry_msgs/PoseStamped 或 nav_msgs/Path 等。
    #
    # 3) 多机器人冲突/碰撞处理：
    #    - 当前解码阶段不处理“多个机器人选择同一个栅格”的冲突，
    #      这种约束完全由 candidate_moves 和 cost 来“间接避免”。
    #    - 完整 TRACE 中可以在 decode 里加入一层 conflict resolution：
    #        * 若多个机器人 next_pos 相同或太近：
    #             - 按优先级/role 让 main 优先，其他机器人回退 candidate_moves 中次优解；
    #             - 或使用一个小的集中式/分布式冲突消解器。
    #
    # 4) 与外层 RL 的接口：
    #    - 现在只是简单地带出 role。
    #    - 完整版可以在 plan 中附加由 RL 决定的调度信息：
    #        * 时间窗长度（若使用多步 horizon）
    #        * 当前调度模式（探索优先/任务优先/通信恢复优先等）
    #      方便后续在真实系统中做日志记录和可视化（例如：给每个 step 打上“调度标签”）。
    #
    # 5) 轨迹缓存与多步执行：
    #    - 当前 decode 只生成“一步”的目标位置。
    #    - 完整 TRACE 中，你可能希望 inner C-ADMM 输出一个
    #      短 horizon 的轨迹 q_i(t:t+H)，此处 decode 的时候将其缓存到每个机器人对象中：
    #         robot.plan_buffer = [(x1,y1), (x2,y2), ...]
    #      外层仿真在每个 step 从 buffer 中取一个执行，从而提高轨迹平滑性与前瞻性。
    # ==============================

    return plan


def _solve_local(rid, z, u_i, env_state, rl_action, link_state, cadmm_params, aux):
    """
    Local solver for each robot i
    for each candidate move point = (nx, ny)
    - construct q_i_candidate = [nx, ny]
    - get r_i(q_i_candidate) = cost
    - get penalty value
    - get local argmin
    """
    # TODO(full-TRACE-q-blocks):
    #   当前 q_i 只包含位置，因此局部子问题只是“在候选位置中选一个点”。
    #   完整 TRACE 中，q_i 需要包含：
    #     - 流量变量 f_{ij}^k(t)（可能是向量）
    #     - 本地任务执行量/服务量变量
    #     - 本地能耗/发射功率等
    #   那时 local subproblem 变成真正的凸优化：
    #       q_i^{k+1} = argmin_{q_i ∈ Q_i} f_i(q_i) + (η/2)||A_i q_i + B_i z + u_i||^2
    #   需要用 PGD / QP / proximal operator 解，而不是枚举 candidate_moves。
    #   建议将“候选动作枚举 + cost 对比”封装`成专门处理位置的模块，
    #   其他 block 用凸优化求解。
    moves = aux["candidate_moves"][rid]
    idx = aux["rid_to_index"][rid]
    z_dim_pos = aux["z_dim_pos"]

    # get global pos for robot i in z
    z_pos = z[:z_dim_pos]
    z_i = z_pos[2* idx: 2* idx + 2]

    best_cost = np.inf
    best_q = None

    for (nx, ny) in moves:
        # 1) candidate q_i
        q_candidate = np.array([float(nx), float(ny)], dtype=float)
        # 2) local cost
        local_cost = _compute_local_cost(rid, q_candidate, env_state, rl_action, link_state, aux)
        # 3) admm penalty
        diff = q_candidate - z_i + u_i
        penalty = 0.5 * cadmm_params.eta * float(np.dot(diff, diff))
        total_cost = local_cost + penalty
        if total_cost < best_cost: 
            best_cost = total_cost
            best_q = q_candidate
    if best_q is None:
        x, y = env_state.robots[rid].pos
        best_q = np.array([float(x), float(y)], dtype=float)
    return best_q

def _compute_local_cost(rid, q_candidate, env_state, rl_action, link_state, aux):
    """
    local cost:
    1) moving cost
    2) COI/explore cost
    3) communication cost
    4) repeat moving cost
    hyper p:
    - cost_move: 0.1
    - gamma = 2.0
    - w_new = 1.0
    - w_fe = 2.0
    -   if role == 1: 
            role_task_weight = 1.0
        elif role == 0: 
            role_task_weight = 0.3
        else:
            role_task_weight = 0.5
    - urgency = 1.0 + 2.0 * (1.0 - normalized_left)

    当前 r_i(q_i) 的最小实现包括：
    - 轨迹相关：移动代价（步长）
    - main：未知区域信息增益 + 熵势场
    - free：COI 距离 + 时间窗 + ddl
    - relay：暂未实装（TODO）
    - 通信：link_sample 未使用（TODO）

    TODO(full-TRACE-ri):
    - 将本文理论中的 r_{i,t}(q_i) 完整拆解为以下部分：
        * r_i^coverage(q_i)   : 信息熵/覆盖相关代价（已经部分实现）
        * r_i^task(q_i)       : 任务执行量/完成率相关代价（现在只通过轨迹近似）
        * r_i^comm(q_i)       : QoS/吞吐/时延相关软约束代价（当前未实现）
        * r_i^energy(q_i)     : 能耗/功率相关代价（当前未实现）
    - 使用与理论部分一致的符号和权重，将这些项与 PPO 的 reward 对齐：
        reward = - Σ_i r_i(q_i)  + 其他全局正则
    - 将当前的启发式写法逐步替换为基于理论模型的公式（例如利用链路容量模型
      C = B log2(1+SNR)，将低 QoS 写成惩罚）。
    """
    # basic information
    robot = env_state.robots[rid]
    x_cur, y_cur = robot.pos
    nx, ny = q_candidate.astype(int)
    role = rl_action.role_assign.get(rid, "free")  # main/free/relay
    t_now = env_state.current_step

    # 1) moving cost
    move_dist = abs(nx - x_cur) + abs(ny - y_cur)
    cost_move = 0.01 * float(move_dist)

    #2) task cost
    cost_task = 0.0
    # explore (main role)
    if role == 0:
        # 主轴代价，需要迫使参与主轴任务的机器人都能够向着未知区域移动 
        r_sense = robot.r_sense
        new_unknown = estimate_new_unknown_cells(env_state, nx, ny, r_sense)

        coverage = compute_current_coverage_ratio(env_state)
        gamma = 2.0
        pressure = 1.0 + gamma*(1.0-coverage)

        w_new = 0.8
        term_new = -w_new * float(new_unknown)

        frontier_entropy = aux.get("frontier_entropy", [])
        phi = entropy_potential_at_pos(nx, ny, frontier_entropy)
        w_fe = 1.0
        term_e = -w_fe * phi
        cost_task += pressure * (term_e + term_new)
        # print(f"rid-{rid} main task: {pressure * (term_e + term_new)}")
        # in future work, add information gain here
    if len(env_state.tasks) != 0:
        if role == 1: 
            role_task_weight = 1.0
        elif role == 0: 
            role_task_weight = 0.3
        else:
            role_task_weight = 0.5
        best_task_cost = None
        for task in env_state.tasks:
            task_id = task.id
            if task.status == "complete" or task.status == "failed": continue
            t_start = task.start_step
            t_ddl = task.ddl_step
            priority = task.priority

            # 2) time limit
            if t_now < t_start or t_now > t_ddl: continue

            # 3) get spatial cost
            tx, ty = task.x, task.y
            d = abs(nx - tx) + abs(ny - ty)

            # 4) time urgency
            T_window = max(1, t_ddl - t_start)
            time_left = max(0, t_ddl - t_now)
            normalized_left = time_left / T_window
            urgency = 1.0 + 2.0 * (1.0 - normalized_left)

            # 5) cost
            base = float(d)
            task_cost = role_task_weight * priority * urgency * base
            if best_task_cost is None or task_cost < best_task_cost:
                best_task_cost = task_cost
        if best_task_cost is not None:
            cost_task += best_task_cost
    # 3) communication cost
    cost_comm = 0.0
    # TODO(后续完整实验):
    #   - 使用 link_sample 中的 RSSI / capacity / plr 等信息，
    #     对 main/relay 施加 QoS 软约束：
    #       * 与上游/下游主轴的 RSSI 低于门限 -> 大罚
    #       * 节点过于拥塞 -> 惩罚
    #   - 可以按角色分开设计 cost_comm_main / cost_comm_relay。
    #4) repulsion cost
    cost_rep = 0.0
    repulsion_grad = aux["repulsion_grad"]
    q_prev_dict = aux["q_prev"]
    if repulsion_grad is not None and q_prev_dict is not None:
        g_i = repulsion_grad[rid]
        q_prev_i = q_prev_dict[rid]
        if g_i is not None and q_prev_i is not None:
            dq = q_candidate - q_prev_i
            w_rep = float(aux["repulsion_weight"])
            cost_rep = w_rep * float(np.dot(g_i, dq))
    total_cost = cost_move + cost_task + cost_comm + cost_rep
    return total_cost
