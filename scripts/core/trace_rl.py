"""
trace_rl.py
Created by JiXX at 20251202
This file is the main function of TRACE
"""
import numpy as np
import networkx as nx
import math; calculate_dist = lambda x1,y1,x2,y2: math.sqrt((x2-x1)**2 + (y2-y1)**2)
from scipy.spatial import cKDTree

import matplotlib.pyplot as plt

from scripts.core.data import CadmmParams, EnvState, RLAction
from scripts.core.cadmm_solver import inner_cadmm
from scripts.environment.robot import UGV, UAV, is_valid_position, random_move_robot
from scripts.environment.grid_map import reveal_with_robot, update_local_maps_based_on_communication
from scripts.environment.communication import update_comm_map, get_link_state, get_main_route
from scripts.environment.task import spawn_random_task
from scripts.utils.visualizer import SimpleVisual
from scripts.utils.loggers import ExplorationLogger
from scripts.utils.helper import shortest_multihop_route_grid, compute_current_coverage_ratio

isVisual=False


class TraceEnv:
    def __init__(self, agent_num: list, env_path):
        self.rng = np.random.default_rng(42)
        self.global_map = np.load(env_path)
        self.dynamic_map = -np.ones_like(self.global_map)
        self.base_pos = [5,5]
        self.robots = []
        # init robot
        num = 0
        init_rl_assign = {}
        for i in range(agent_num[0]):
            self.robots.append(UGV(id=i))
            num += 1
        for i in range(num, num + agent_num[1]):
            self.robots.append(UAV(id=i))
        init_q = {}
        for robot in self.robots:
            x, y = self.sample_positin_for(robot)
            robot.update_pos(x, y)
            robot.local_map = -np.ones_like(self.global_map)
            init_rl_assign[robot.id] = 0
            init_q[robot.id] = (x, y)
            robot.get_name()
        # init comm map
        self.comm_map = nx.DiGraph()
        for robot in self.robots:
            self.comm_map.add_node(f"{robot.type_id}_{robot.id}", pos=robot.pos)
        self.comm_map.add_node("base", pos=self.base_pos)
        # update local map for each robot
        for robot in self.robots:
            reveal_with_robot(self.global_map, self.dynamic_map, robot)
        update_comm_map(self.global_map, self.comm_map, self.robots)
        update_local_maps_based_on_communication(self.comm_map, self.robots)

        # init task
        self.tasks = []
        self.consec_steps = 1
        # init inner cadmm
        self.env_state = EnvState(
            global_map=self.global_map.copy(), 
            explored_map=self.dynamic_map,
            base_pos=self.base_pos,
            robots=self.robots,
            tasks=self.tasks, 
            t_n=0, 
            window_horizon=1,
            current_step=0
        )
        self.rl_action = RLAction(
            role_ratio=np.array([1, 1]), 
            role_assign=init_rl_assign
        )
        self.link_state = get_link_state(comm_map=self.comm_map)
        # TODO: 后续这个参数需要放置在cfg文件中进行统一管理
        self.cadmm_params = CadmmParams(
            eta=0.1,
            alpha=0.1,
            eps_pri=0.1,
            eps_dual=0.1,
            k_max=20,
            ttl_hops=3,
            t_fresh=3,

            move_w=0.05,
            cover_gamma=2.5,
            new_w=0.5,
            fe_w=0.75,

            role_main_weight=1.0,
            role_relay_weight=0.5,
            role_task_weight=0.2,

            qos_default_weight=0.1,
            qos_exec_weight=1.0,
            qos_relay_weight=1.0,

            qos_improve_w=1.0, 
            qos_break_w=0.2, 
            qos_degrade_w=1.0,

            urgency_w=1.0,

            rep_w=0.5,
            rep_sigma=2.0
        )
        # init visual
        if isVisual:
            self.visual = SimpleVisual(self.robots, self.dynamic_map, self.comm_map)

        # init log
        self.logger = ExplorationLogger(self.robots)

    def update(self, step, visible_cells_this_step ):
        self.env_state.current_step = step
        # 这里替换为inner cadmm设计的plan
        inner_plan = inner_cadmm(self.env_state, self.rl_action, self.link_state, self.cadmm_params)
        for robot_id, plan in inner_plan.robot_plans.items():
            new_x, new_y = plan["next_pos"][0], plan["next_pos"][1]
            self.robots[robot_id].update_pos(new_x, new_y)
        update_comm_map(self.global_map, self.comm_map, self.robots)
        current_qos = get_link_state(comm_map=self.comm_map)
        # 根据这个comm map找到一个能够连通的主轴图
        for robot in self.robots:
            reveal_with_robot(self.global_map, self.dynamic_map, robot, visible_cells_this_step )
        update_local_maps_based_on_communication(self.comm_map, self.robots)
        self.logger.log_step(step, self.global_map, self.dynamic_map, self.comm_map)
        # 每次执行完动作后更新env_state, link_state
        

    def update_task(self, step, visible_cells_this_step: set):
        """
        根据当前时间步 robots 的可见格子集合，更新任务状态：
        - 若 (task.x, task.y) 在 visible_cells_this_step 中，则 consec_seen += 1
        - 若不在，则 consec_seen 归零
        - consec_seen >= 2 时，任务完成
        - 当前时间 > deadline 且还未完成时，任务失败
        同时，当任务初次出现时，依据简单的距离进行节点rl_action中的role_assign判定，每次判定都只会执行一次，避免抖动
        - 距离最近的节点作为task的initRid
            * task在感知范围内
                % 使用shortest_multihop_route_grid函数计算从initRid节点到base的最优relay路径
                % 根据距离将除了base和initRid位置的其余中继位置分配给当前正在执行main任务的节点，这些节点的任务状态切换为relay
            * task不在感知范围内
                % initRid对应的节点任务状态切换为free
        """
        for task in self.tasks:
            if task.status == "completed" or task.status == "failed": 
                for robot in self.robots:
                    if task.id in robot.sequence: del robot.sequence[task.id]
                continue

            if step > task.ddl_step and task.status != "completed":
                task.status = "failed"
                for robot in self.robots:
                    if task.id in robot.sequence: del robot.sequence[task.id]
                continue
            # if (task.x, task.y) in visible_cells_this_step:  # 原先判据
            x_t, y_t = task.x, task.y
            paths = get_main_route(self.comm_map, self.robots, task)
            if len(paths) != 0: 
                task.consec_seen += 1
                if task.consec_seen >= self.consec_steps and task.status != "completed":
                    task.status = "completed"
                    task.complete_step = step
                    self.global_map[task.y][task.x] = 3
                else:
                    if task.status == "pending": task.status = "observed"
            else:
                task.consec_seen = 0
                if task.status == "observed":
                    pass
            if task.initRid is None:
                min_dist = np.inf
                min_rid = None
                cand_rid = None
                for rid, robot in enumerate(self.robots):
                    curr_x, curr_y = robot.pos
                    dist = calculate_dist(curr_x, curr_y, x_t, y_t)
                    if min_dist > dist:
                        min_dist = dist
                        min_rid = robot.name
                        cand_rid = rid
                if min_rid:
                    task.initRid = cand_rid
                    # 添加从rid 到 base的中继路径
                    self.robots[cand_rid].sequence[task.id] = [np.array([x_t, y_t], dtype=int), 2]
            if task.initRid is not None:               
                curr_x, curr_y = self.robots[task.initRid].pos
                dist = calculate_dist(curr_x, curr_y, x_t,y_t)
                if task.ref_path is None and dist < self.robots[task.initRid].r_sense:
                    task.ref_path = shortest_multihop_route_grid(self.global_map, (curr_x, curr_y), (self.base_pos[0], self.base_pos[1]), threshold=3.0, RANGE=25)
                    # 依据绝对距离判断ref_path中每个中继位置分配给哪个robot
                    robots_pos = np.asarray(
                        [[robot.pos[0], robot.pos[1]] for robot in self.robots], dtype=float
                    )
                    mids = task.ref_path[1:-1]
                    points = np.asarray(mids, dtype=float)
                    if len(mids) != 0:
                        tree = cKDTree(robots_pos)
                        _, nn = tree.query(points, k=1)
                        assigned_rids = nn.tolist()
                        for rid, pt in zip(nn.tolist(), mids):
                             self.robots[rid].sequence[task.id] = [np.array([pt[0], pt[1]], dtype=int), 1]
                        # 2) 构造 tx->...->rx 的 robot 有序链（用于后续知道前后是谁）
                        #    注：中继点可能被同一机器人连续分配，你可以选择保留或压缩
                        full_chain = [task.initRid] + assigned_rids + ["base"]

                        # 可选：压缩连续重复（更像“链路节点序列”，避免同一 robot 连续出现多次）
                        compact_chain = []
                        for rid in full_chain:
                            if not compact_chain or rid != compact_chain[-1]:
                                compact_chain.append(rid)

                        task.robot_chain = compact_chain  # 有序序列

                        # 3) 生成每个链上节点的 prev/next，方便 O(1) 查询邻居
                        neighbors = {}
                        for i, rid in enumerate(task.robot_chain):
                            prev_r = task.robot_chain[i - 1] if i > 0 else None
                            next_r = task.robot_chain[i + 1] if i < len(task.robot_chain) - 1 else None
                            neighbors[rid] = {"prev": prev_r, "next": next_r}

                        task.chain_neighbors = neighbors    # task.chain_neighbors[rid]["prev"/"next"]
                        print(f"Task {task.id} route: {task.robot_chain}")
    # utils
    def sample_positin_for(self, robot):
        for _ in range(1000):
            x = int(self.rng.integers(self.global_map.shape[0]))
            y = int(self.rng.integers(self.global_map.shape[1]))
            if x > 10 or y > 10: continue
            if is_valid_position(self.global_map, robot, x, y):
                if all(not (r.pos[0] == x and r.pos[1] == y) for r in self.robots):
                    return x, y
        raise RuntimeError("Cannot find valid initial positions")
    
if __name__ == "__main__":
    traceEnv = TraceEnv([3, 3], "D:\研究生\TJU\工作情况\9 大论文\实验部分\grid地图最小验证\scripts\environment\gridMap80.npy")
    num_steps = 50
    visible_cells_this_step = set()
    for step in range(num_steps):
        if isVisual:
            traceEnv.visual.update(traceEnv.comm_map, traceEnv.dynamic_map, traceEnv.robots)
        # if traceEnv.rng.integers(10) > 5 and step < 25:
        if step % 5 == 0:
            spawn_random_task(traceEnv.global_map, traceEnv.dynamic_map, traceEnv.tasks, step, traceEnv.rng)
        traceEnv.update(step, visible_cells_this_step )
        # 这里任务更新，需要加入一个判断，为每个task选择最近的节点作为发起节点
        traceEnv.update_task(step, visible_cells_this_step)
        print(f"============= Step {step} ==============")
        for robot in traceEnv.robots:
            print(robot.sequence)
    num_tasks = len(traceEnv.tasks)
    num_success = 0
    for task in traceEnv.tasks:
        if task.status == "completed": num_success += 1
        print(task)
    print(f"Final cover rate is : {compute_current_coverage_ratio(traceEnv.env_state)}")
    print(f"Final success rate is {num_success / num_tasks:.2f}")
    plt.ioff()
    plt.show()

            