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

Update:
Inner solver 不应该依赖外部可变对象（env_state、robot 类实例），而应该只依赖一个 CadmmProblem 快照
"""
from dataclasses import dataclass
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import defaultdict

from scripts.core.data import CadmmParams, CadmmWarmStart, EnvState, InnerPlan, RLAction, BlockSpec, LocalQ, LocalU
from scripts.environment.communication import LinkState, estimate_link_state
from scripts.environment.robot import UAV, UGV, enumerate_candidate_moves
from scripts.environment.grid_map import estimate_new_unknown_cells, compute_frontier_entropy, predict_coverage_after_moves
from scripts.utils.helper import compute_current_coverage_ratio, entropy_potential_at_pos, update_repulsion_grad, role_priority
# new added at 20260202
from scripts.core.data import FeatureFlags, CadmmProblem, CadmmWarmStart, CadmmDiagnostics, InnerSolution



# ==== tools ====
def _get_z_block(z: np.ndarray, aux, name: str):
    spec = aux["z_blocks"][name]
    return z[spec.start:spec.end].reshape(spec.shape)

def _set_z_block(z: np.ndarray, aux, name: str, value: np.ndarray):
    spec = aux["z_blocks"][name]
    z[spec.start:spec.end] = np.asarray(value).reshape(-1)

def _prox_pos_block(c_pos: np.ndarray, env_state, cadmm_params, rl_action: RLAction, aux) -> np.ndarray:
    """
    对位置 block 做投影：
      1) 把连续 c_pos 映射到最近的 grid integer；
      2) 若多个机器人抢同一格，按角色/ID优先级选一个胜者，其余退回当前格或最近备选格；
    c_pos: shape (num_robots, 2)
    返回: z_pos_proj, 同 shape
    """
    robot_ids = aux["robot_ids"]
    num_robots = len(robot_ids)
    id2idx = aux["rid_to_index"]

    z_int = np.rint(c_pos).astype(int)
    cell_to_rids = {}
    for rid in robot_ids:
        idx = id2idx[rid]
        x, y = z_int[idx]
        cell = (int(x), int(y))
        cell_to_rids.setdefault(cell, []).append(rid)

    roles = {}
    for rid in robot_ids:
        roles[rid] = rl_action.role_assign.get(rid, "free")

    occupied_cells = {}
    z_proj = z_int.copy()
    for cell, rids in cell_to_rids.items():
        if len(rids) == 1:
            occupied_cells[cell] = rids[0]
            continue
        winner = max(
            rids, 
            key=lambda rid: (role_priority(roles[rid]), -rid)
        )
        occupied_cells[cell] = winner
        for rid in rids:
            if rid == winner: continue
            idx = id2idx[rid]
            x_cur, y_cur = env_state.robots[rid].pos
            x_cur, y_cur = int(x_cur), int(y_cur)
            z_proj[idx] = np.array([x_cur, y_cur], dtype=int)

    return z_proj.astype(int)


def _prox_cov_block(c_cov: np.ndarray, env_state, cadmm_params, aux) -> np.ndarray:
    """
    覆盖率 block 的 prox。
    当前版本：简单 clip 到 [0, 1] 区间。
    """
    return np.clip(c_cov, 0.0, 1.0)


def _prox_flow_block(c_flow: np.ndarray, env_state, cadmm_params, aux) -> np.ndarray:
    """
    flow block 的占位 prox。
    当前版本：identity，将来可以加容量/守恒等约束投影。
    """
    return c_flow


def _prox_qos_block(c_qos: np.ndarray, env_state, cadmm_params, aux) -> np.ndarray:
    """
    qos block 的占位 prox。
    当前版本：identity，将来可以加 QoS 硬约束（如带宽/时延）投影。
    """
    return c_qos


def _prox_energy_block(c_energy: np.ndarray, env_state, cadmm_params, aux) -> np.ndarray:
    """
    energy block 的占位 prox。
    当前版本：identity，将来可以加入 [0, E_max] 的 box 投影等。
    """
    return c_energy


def _prox_task_block(c_task: np.ndarray, env_state, cadmm_params, aux) -> np.ndarray:
    """
    task block 的占位 prox。
    当前版本：identity，将来可以加入 Σ_i x_{i,l} = 1 等任务守恒投影。
    """
    return c_task

# === tools ====


# TODO: 按照统一接口进行函数调用
# def inner_cadmm_entry(
#     env_state,             # 外部对象，只用于构造 problem snapshot
#     link_state_current,
#     cadmm_params: CadmmParams,
#     flags: FeatureFlags,
#     warm_start: CadmmWarmStart | None,
#     step: int,
#     rng,
# ) -> tuple[InnerSolution, CadmmWarmStart, CadmmDiagnostics]:
#     problem = build_problem_snapshot(...)
#     return solve_inner_cadmm(problem, warm_start)


# # Global cadmm entrance added by JiXX at 20260202
# def solve_inner_cadmm(problem:CadmmProblem, warm_start: CadmmWarmStart | None = None) -> Tuple["InnerSolution", CadmmWarmStart, CadmmDiagnostics]:
#     ...


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
    q, z, u, aux = _init_cadmm_state(env_state, cadmm_params, rl_action)

    q_current = q.copy()
    # Step 2 Main optimize loop
    r_hist, s_hist = [], []
    for k in range(cadmm_params.k_max):
        # update q for each robot
        aux["q_prev"] = {idx: q_current[idx].copy() for idx in aux["robot_ids"]}
        aux["rep_sigma"] = cadmm_params.rep_sigma
        update_repulsion_grad(aux, q_current)

        q_new = _local_update_all(q_current, z, u, env_state, rl_action, link_state, cadmm_params, aux)
        # global update z
        z_new = _global_update_z(q_new, z, u, env_state, cadmm_params, rl_action, aux)

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

def _init_cadmm_state(env_state: EnvState, cadmm_params, rl_action=None, warm_start=None):
    """
    Initialize ADMM state, including: local q, global z, dual u and helper aux
    - q: q_i denotes next step pos for each robot
    - z: cat of positions and global coverage ratio
    - u: 0
    TODO: add link state, QoS state to z
    """
    robot_ids = [robot.id for robot in env_state.robots]
    num_robots = len(robot_ids)

    # 20251211新增, 为全局变量z添加符合理论值的blocks
    blocks = {}
    offset = 0
    dim_pos = 2 * num_robots
    # 20251211新增, pos block, 
    # 20260202修改为新的block定义形式
    blocks["pos"] = BlockSpec(
        name="pos", 
        start=offset, 
        end=offset+dim_pos, 
        shape=(num_robots, 2)
    )
    offset += dim_pos
    # 20251211新增, cover block
    dim_cov = 1
    blocks["cov"] = BlockSpec(
        name="cov", 
        start=offset, 
        end=offset+dim_cov,
        shape=(dim_cov,)
    )
    offset += dim_cov
    # # 20251211新增, 其他block后续陆续添加，这里只进行占位设计
    # dim_flow = 1
    # blocks["flow"] = BlockSpec(
    #     name="flow", 
    #     start=offset,
    #     end=offset+dim_flow, 
    #     shape=(dim_flow,)
    # )
    # offset += dim_flow

    # dim_qos = 1
    # blocks["qos"] = BlockSpec(
    #     name="qos", 
    #     start=offset,
    #     end=offset+dim_qos, 
    #     shape=(dim_qos,)
    # )
    # offset += dim_qos

    # dim_energy = num_robots
    # blocks["energy"] = BlockSpec(
    #     name="energy",
    #     start=offset,
    #     end=offset+dim_energy,
    #     shape=(dim_energy, )
    # )
    # offset += dim_energy

    # dim_tasks = 1
    # blocks["tasks"] = BlockSpec(
    #     name="tasks",
    #     start=offset,
    #     end=offset+dim_tasks,
    #     shape=(dim_tasks,)
    # )
    # offset += dim_tasks
    z_dim = offset

    # 1) get action candidate for each robot
    candidate_moves: Dict[int, List[tuple[int, int]]] = {}  # id: [x, y]
    for rid in robot_ids:
        candidate_moves[rid] = enumerate_candidate_moves(env_state, rid)

    frontier_entropy = compute_frontier_entropy(env_state)
    # print(frontier_entropy)
    aux: Dict[str, Any] = {
        "robot_ids": robot_ids,
        "num_robots": num_robots,
        "candidate_moves": candidate_moves,
        "rid_to_index": {rid: idx for idx, rid in enumerate(robot_ids)},
        "z_dim_pos": blocks["pos"].end - blocks["pos"].start,
        "z_dim_cov": blocks["cov"].end - blocks["cov"].start, 
        # 完整版TRACE中，z所需要包含的blocks和对应形状及索引
        "z_blocks": blocks,
        "z_dim": z_dim,
        "frontier_entropy": frontier_entropy, 
        "q_prev": {idx: env_state.robots[idx].pos for idx in robot_ids}, 
        "repulsion_grad": {idx: np.zeros(2, dtype=float) for idx in robot_ids}, 
        "rep_sigma": 2.0,
    }

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
                q[rid] = LocalQ(
                    pos=np.array([float(x), float(y)], dtype=float)
                )
            if rid in warm_start.u_prev:
                u[rid] = warm_start.u_prev[rid].copy()
            else:
                u[rid] = LocalU(
                    pos=np.zeros(2, dtype=float)
                )
        expected_dim = z_dim
        if warm_start.z_prev.shape[0] == expected_dim:
            z = warm_start.z_prev.copy()
        else:
            z = np.zeros(z_dim, dtype=float)
            z_pos_coords: List[float] = []
            for rid in robot_ids:
                x, y = env_state.robots[rid].pos
                z_pos_coords.extend([float(x), float(y)])
            z_pos = np.array(z_pos_coords, dtype=float)
            z_cov = compute_current_coverage_ratio(env_state)
            _set_z_block(z, aux, "pos", z_pos)
            _set_z_block(z, aux, "cov", np.array([z_cov], dtype=float))
    else:
        # ---- cold start ----
        z_pos_coords: List[float] = []
        z = np.zeros(z_dim, dtype=float)
        for rid in robot_ids:
            x, y = env_state.robots[rid].pos
            q[rid] = LocalQ(
                pos=np.array([float(x), float(y)], dtype=float)
            )
            u[rid] = LocalU(
                pos=np.zeros(2, dtype=float)
            )
            z_pos_coords.extend([float(x), float(y)])

        z_pos = np.asarray(z_pos_coords, dtype=float)
        z_cov = compute_current_coverage_ratio(env_state)
        _set_z_block(z, aux, "pos", z_pos)
        _set_z_block(z, aux, "cov", np.array([z_cov], dtype=float))

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
            q_i=q[rid],
            z=z,
            u_i=u[rid],
            env_state=env_state,
            rl_action=rl_action,
            link_state=link_state,
            cadmm_params=cadmm_params,
            aux=aux
        )
    return new_q

# TODO 按照更新后的dataclass更新函数
# def update_z(
#     problem: CadmmProblem,
#     q: np.ndarray,    # (N,D)
#     u: np.ndarray,    # (N,D)
#     z_prev: np.ndarray # (D,)
# ) -> tuple[np.ndarray, dict[str,float]]:  # z_new, violation_by_block


def _global_update_z(q, z_old, u, env_state, cadmm_params, rl_action, aux) -> np.ndarray:
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
    blocks = aux["z_blocks"]
    z_new = z_old.copy()
    alpha = cadmm_params.alpha
    # TODO(full-TRACE-overrelaxation):
    #   当前仅对位置 block 做简单的过松弛：
    #       z_pos_relaxed = α z_pos_new + (1-α) z_pos_old
    #   完整版需要：
    #     - 定义 c^{k+1} = α z^{k+1} + (1-α) z^k 并在 q-步中使用 c 而非 z；
    #     - 对异步时变图，引入混合矩阵 W(k) 对 c 做邻居聚合；
    #     - 将这些实现与理论部分的“过松弛 & 图适配”定理对应起来。

    # 1) position block
    if "pos" in blocks:
        c_pos = np.zeros((num_robots, 2), dtype=float)
        for rid in robot_ids:
            idx = aux["rid_to_index"][rid]
            c_pos[idx, :] = q[rid].pos + u[rid].pos
        # prox
        z_pos_prox = _prox_pos_block(c_pos, env_state, cadmm_params, rl_action, aux)
        z_pos_old = _get_z_block(z_old, aux, "pos")
        if alpha != 1.0:
            z_pos_use = alpha * z_pos_prox + (1.0 - alpha) * z_pos_old
        else:
            z_pos_use = z_pos_prox
        _set_z_block(z_new, aux, "pos", z_pos_use)

    # 2) cov block
    if "cov" in blocks:
        cover_scalar = predict_coverage_after_moves(env_state, q)
        c_cov = np.array([cover_scalar], dtype=float)
        # prox
        z_cov_prox = _prox_cov_block(c_cov, env_state, cadmm_params, aux)
        _set_z_block(z_new, aux, "cov", z_cov_prox)

    # 后续添加相关定义函数后添加
    # 3) flow block
    # 4) qos block
    # 5) energy block
    # 6) task block

    return z_new
    
# TODO 按照更新后的dataclass进行函数更新
# def update_u(problem, q, z, u, eta) -> np.ndarray:
# def compute_residuals(problem, q, z, z_prev) -> tuple[dict[str,float], dict[str,float]]:s
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
    blocks = aux["z_blocks"]

    eta = cadmm_params.eta
    eps_pri = cadmm_params.eps_pri
    eps_dual = cadmm_params.eps_dual

    # 1) get block for new and previous pos
    spec_pos = blocks["pos"]
    z_pos_new = z_new[spec_pos.start:spec_pos.end].reshape(num_robots, 2)
    z_pos_old = z[spec_pos.start:spec_pos.end].reshape(num_robots, 2)

    # 2) original residual r = q - z_pos_new, update dual parameter u
    r_list = []
    new_u = {}
    for rid in robot_ids:
        idx = aux["rid_to_index"][rid]

        z_i_new = z_pos_new[idx]
        q_i  = q[rid]
        r_i = q_i.pos - z_i_new
        r_list.extend(r_i.tolist())

        u_i_old = u[rid]
        u_i_new = u_i_old.pos + r_i
        new_u[rid] = LocalU(
            pos = u_i_new
        )
    
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

    目前的冲突消解策略是根据节点的角色进行评价，执行主轴任务的优先级最高、中继次之、自由任务优先级最低，
    根据优先级高低决定cell归属，如果优先级一样，按照robot id先后决定cell归属
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
        x_target_f = float(q_i.pos[0])
        y_target_f = float(q_i.pos[1])
        moves = candidate_moves[rid]
        best_move = None
        best_d2 = np.infty
        for (cx, cy) in moves:
            dx = cx - x_target_f
            dy = cy - y_target_f
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                second_move, second_d2 = best_move, best_d2
                best_move, best_d2 = (cx, cy), d2
            elif d2 < second_d2:
                second_move, second_d2 = (cx, cy), d2
        if best_move is None:
            best_move = cur_pos[rid]
            second_move = None
        proposals[rid] = {
            "primary": best_move,
            "secondary": second_move,
            "chosen": best_move
        }
        cell_to_rids[best_move].append(rid)
    
    for cell, rids in list(cell_to_rids.items()):
        if len(rids) <= 1: continue
        winner = max(
            rids, key=lambda rid: (role_priority(roles[rid]), -rid)
        )
        for rid in rids:
            if rid == winner: continue
            second = proposals[rid]["secondary"]
            old_cell = proposals[rid]["chosen"]

            if second is not None and (second not in cell_to_rids or len(cell_to_rids[second]) == 0):
                proposals[rid]["chosen"] = second
                cell_to_rids[old_cell].remove(rid)
                cell_to_rids[second].append(rid)
            else:
                stay_cell = cur_pos[rid]
                proposals[rid]["chosen"] = stay_cell
                cell_to_rids[old_cell].remove(rid)
                cell_to_rids[stay_cell].append(rid)

    for i, rid_i in enumerate(robot_ids):
        for rid_j in robot_ids[i+1:]:
            ci = cur_pos[rid_i]
            cj = cur_pos[rid_j]
            ni = proposals[rid_i]["chosen"]
            nj = proposals[rid_j]["chosen"]
            if ni == cj and nj == ci:
                pri_i = role_priority(roles[rid_i])
                pri_j = role_priority(roles[rid_j])
                if pri_i > pri_j:
                    loser = rid_j
                elif pri_i < pri_j:
                    loser = rid_i
                else:
                    loser = rid_j if rid_j > rid_i else rid_i
                proposals[loser]["chosen"] = cur_pos[loser]

    plan = {}
    for rid in robot_ids:
        nx, ny = proposals[rid]["chosen"]
        x_cur, y_cur = cur_pos[rid]
        dx = nx - x_cur
        dy = ny - y_cur
        role = roles[rid]
        plan[rid] = {
            "rid": rid, 
            "role": role, 
            "next_pos": (nx, ny), 
            "delta": (dx, dy), 
            "stay": (dx==0 and dy==0),
            "action_type": "move" if (dx != 0 or dy != 0) else "hold"
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

# TODO: 更新本地q值更新与新的dataclass对齐
# def solve_local_q(
#     rid: int,
#     problem: CadmmProblem,
#     z: np.ndarray,      # (D,)
#     u_i: np.ndarray,    # (D,)
#     q_i_prev: np.ndarray, # (D,) for warm local
# ) -> np.ndarray:        # (D,) returns q_i
def _solve_local(rid, q_i: LocalQ, z, u_i: LocalU, env_state, rl_action, link_state, cadmm_params, aux):
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
    eta = cadmm_params.eta
    moves = aux["candidate_moves"][rid]

    best_cost = np.inf
    best_q = None

    robot_ids = aux["robot_ids"]
    block = aux["z_blocks"]
    spec_pos = block["pos"]
    z_pos_vec = z[spec_pos.start:spec_pos.end]
    idx = aux["rid_to_index"][rid]

    # get global pos for robot i in z
    z_i = z_pos_vec[2* idx: 2* idx + 2]
    u_i_pos = u_i.pos

    for (nx, ny) in moves:
        # 1) candidate q_i
        q_pos_cand = np.array([float(nx), float(ny)], dtype=float)
        q_candidate = LocalQ(
            pos=q_pos_cand
        )
        # 2) local cost
        local_cost = _compute_local_cost(rid, q_candidate, env_state, rl_action, link_state, aux, cadmm_params)
        # 3) admm penalty
        diff = q_pos_cand - z_i + u_i.pos
        penalty = 0.5 * cadmm_params.eta * float(np.dot(diff, diff))
        total_cost = local_cost + penalty
        if total_cost < best_cost: 
            best_cost = total_cost
            best_q = LocalQ(
                pos=q_pos_cand
            )
    if best_q is None:
        best_q = q_i.copy()

    return best_q

def _compute_local_cost(rid, q_candidate: LocalQ, env_state: EnvState, rl_action: RLAction, link_state: LinkState, aux, cadmm_params):
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
    nx, ny = q_candidate.pos.astype(int)
    role = rl_action.role_assign.get(rid, 0)  # main/free/relay
    t_now = env_state.current_step

    # ---------------- 1) moving cost ----------------
    move_dist = abs(nx - x_cur) + abs(ny - y_cur)
    cost_move = cadmm_params.move_w * float(move_dist)

    # =================================================
    # 2) explore cost：覆盖 / 前沿（无任务时主导，有任务时减弱）
    # =================================================
    cost_explore = 0.0
    seq = getattr(robot, "sequence", {})
    has_seq = bool(seq)
    # 主轴代价，需要迫使参与主轴任务的机器人都能够向着未知区域移动 
    r_sense = robot.r_sense
    new_unknown = estimate_new_unknown_cells(env_state, nx, ny, r_sense)

    coverage = compute_current_coverage_ratio(env_state)
    pressure = 1.0 + cadmm_params.cover_gamma*(1.0-coverage)

    term_new = -cadmm_params.new_w * float(new_unknown)

    frontier_entropy = aux.get("frontier_entropy", [])
    phi = entropy_potential_at_pos(nx, ny, frontier_entropy)
    term_e = -cadmm_params.fe_w * phi
    base_explore = pressure * (term_e + term_new)
    if not has_seq:
        explore_weight = 1  # cadmm_params.explore_idle_weight
    else:
        explore_weight = 0.01
    cost_explore = explore_weight * base_explore

    # =================================================
    # 3) sequence-based task cost：COI 执行 + 中继节点靠近目标点
    # =================================================
    cost_task_seq = 0.0
    best_seq_cost = None
    if seq:
        for task in env_state.tasks:
            tid = task.id
            if tid not in seq: continue
            if task.status in ("completed", "failed"): continue

            target_pos, ttype = seq[tid]
            tx, ty = int(target_pos[0]), int(target_pos[1])
            # time window
            t_start = task.start_step
            t_ddl = task.ddl_step
            if t_now < t_start or t_now > t_ddl: continue

            T_window = max(1, t_ddl - t_start)
            time_left = max(0, t_ddl - t_now)
            normalized_left = time_left / T_window
            urgency = 1.0 + cadmm_params.urgency_w * (1.0 - normalized_left)
            priority = getattr(task, "priority", 1.0)

            d = abs(nx - tx) + abs(ny - ty)
            if ttype == 2:  # coi executor
                w_type = cadmm_params.role_task_weight
            elif ttype == 1:  # relay
                w_type = cadmm_params.role_relay_weight
            else:  # main
                w_type = cadmm_params.role_main_weight
            base = float(d)
            seq_cost = w_type * priority * urgency * base
            if best_seq_cost is None or seq_cost < best_seq_cost:
                best_seq_cost = seq_cost
        if best_seq_cost is not None:
            cost_task_seq += best_seq_cost
            
    # =================================================
    # 4) QoS / chain cost：基于 chain_neighbors，检查“前后链路是否变好”
    # =================================================
    cost_qos = 0.0
    if seq:
        for task in env_state.tasks:
            tid = task.id
            if tid not in seq: continue
            if task.status in ("completed", "failed"): continue

            chain = getattr(task, "chain_neighbors", None)
            if chain is None: continue
            # time window
            t_start = task.start_step
            t_ddl = task.ddl_step
            if t_now < t_start or t_now > t_ddl: continue

            T_window = max(1, t_ddl - t_start)
            time_left = max(0, t_ddl - t_now)
            normalized_left = time_left / T_window
            urgency = 1.0 + cadmm_params.urgency_w * (1.0 - normalized_left)
            priority = getattr(task, "priority", 1.0)
            _, ttype = seq[tid]
            if ttype == 2:
                axis_role_weight = cadmm_params.qos_exec_weight
            elif ttype == 1:
                axis_role_weight = cadmm_params.qos_relay_weight
            else:
                axis_role_weight = cadmm_params.qos_default_weight
            nb_info = chain[rid]
            neighbours_ids = []
            if nb_info.get("prev", None) is not None:
                neighbours_ids.append(nb_info["prev"])
            if nb_info.get("next", None) is not None:
                neighbours_ids.append(nb_info["next"])
            thr = 3.0
            link_cost_sum = 0.0
            num_links = 0
            for nb in neighbours_ids:
                if nb == "base":
                    pos_nb = np.array(env_state.base_pos, dtype=float)
                else:
                    if nb < 0 or nb > len(env_state.robots):
                        continue
                    pos_nb = env_state.robots[nb].pos
                pos_cur = np.array([x_cur, y_cur], dtype=float)
                pos_new = np.array([nx, ny], dtype=float)
                sig_cur = estimate_link_state(env_state, pos_cur, pos_nb)
                sig_new = estimate_link_state(env_state, pos_new, pos_nb)
                if sig_new < thr:
                    link_cost = cadmm_params.qos_break_w * float(thr - sig_new)
                else:
                    link_cost = 0.0
                if sig_new > sig_cur:
                    link_cost += -cadmm_params.qos_improve_w * float(sig_new - sig_cur)
                elif sig_new < sig_cur:
                    link_cost += cadmm_params.qos_degrade_w * float(sig_new - sig_cur)
                link_cost_sum += link_cost
                num_links += 1
            if num_links > 0:
                avg_link_cost = link_cost_sum / float(num_links)
                cost_qos += axis_role_weight * priority * urgency * avg_link_cost

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
            dq = q_candidate.pos - q_prev_i.pos
            cost_rep = cadmm_params.rep_w * float(np.dot(g_i, dq))
    total_cost = cost_move + cost_explore + cost_task_seq + cost_qos + cost_rep
    return total_cost
