"""
grid_map.py
Created by JiXX at 20251202
- Define how robot reveals unknown map:
    * reveal_with_robot: reveal global dynamic map and robot local map according to robot position and sensing radius
"""
import networkx as nx
import numpy as np

from scripts.utils.helper import has_line_of_sight


def reveal_with_robot(global_map, dynamic_map, robot, visible_this_step=None):
    """
    根据机器人的感知范围和视线，在 dynamic_map 上揭露地图：
    - dynamic_map 初值为 -1（未知）
    - 若某格在感知半径内且有视线，则 dynamic_map[y,x] = global_map[y,x]
    """
    h, w = global_map.shape
    x0, y0 = robot.pos

    r_sense = robot.r_sense
    for dx in range(-r_sense, r_sense):
        for dy in range(-r_sense, r_sense):
            nx, ny = x0 + dx, y0 + dy
            if not(0 <= nx < w and 0 <= ny < h): continue
            if max(abs(dx), abs(dy)) > r_sense: continue
            isLine = has_line_of_sight(global_map, x0, y0, nx, ny)
            if isLine:
                robot.local_map[ny, nx] = global_map[ny, nx]
                dynamic_map[ny, nx] = global_map[ny, nx]
                if visible_this_step is not None:
                    visible_this_step.add((nx, ny))

def update_local_maps_based_on_communication(comm_map: nx.DiGraph, robots):
    """
    根据全局通信图，更新机器人的局部地图
    - 如果存在信号强度 signal > 3 的边，则合并其他机器人局部地图的信息
    """
    # 遍历所有边，检查信号强度
    for robot1 in robots:
        # 机器人1的局部地图
        for robot2 in robots:
            if robot1 == robot2:
                continue  # 跳过自己
            
            # 获取从robot2到robot1的边，查看信号强度
            edge_signal = comm_map.get_edge_data(f"{robot2.type_id}_{robot2.id}", f"{robot1.type_id}_{robot1.id}")
            if edge_signal is None:
                continue  # 如果没有从robot2到robot1的边，跳过
            
            signal = edge_signal.get('signal', 0)
            
            if signal > 3:
                robot1.local_map[(robot1.local_map==-1)&(robot2.local_map!=-1)] = robot2.local_map[(robot1.local_map==-1)&(robot2.local_map!=-1)]
    
def estimate_new_unknown_cells(env_state, nx, ny, r_sense):
    global_map = env_state.global_map
    explored_map = env_state.explored_map
    H, W = global_map.shape
    count = 0
    for dx in range(-r_sense, r_sense):
        for dy in range(-r_sense, r_sense):
            cx, cy = nx + dx, ny + dy
            if not(0 <= cx < W and 0 <= cy < H): continue
            if explored_map[cy, cx] == -1: 
                count += 1
    return count


def compute_frontier_entropy(env_state):
    """
    计算所有前沿未知单元的列表及其熵值 H_f。

    最小实现：
    - unknown 定义为: global_map == 0 且 explored_map == 0
    - frontier 定义为: unknown 且 4 邻域中至少有一个已探索自由格
    - 熵 H_f 暂统一为 1.0（对应 p=0.5 最大熵）

    返回:
    - 列表 [ (fx, fy, H_f), ... ]

    TODO(full-TRACE-entropy-field):
    - 当前简化为：每个 frontier cell 的熵 H_f = 1.0。
    - 完整版需要：
      1) 使用 occupancy 概率 p(y,x) 计算 H_f:
           H_f = -p log p - (1-p) log(1-p)
         其中 p 由贝叶斯更新/传感器模型给出；
      2) 区分不同类型未知：例如“高不确定区域” vs “已多次观测仍不确定区域”；
      3) 将 main 的熵势场与 MEF-Explore 中定义的 entropy field 对齐：
         即：Φ(p) ≈ ∫_Ω H(x) K(p,x) dx 或其离散近似。
    """
    global_map = env_state.global_map
    explored_map = env_state.explored_map
    H, W = global_map.shape

    frontier_list = []

    for y in range(H):
        for x in range(W):
            # 未知 & 可行走
            if not (global_map[y, x] == 0 and explored_map[y, x] == -1):
                continue
            # 检查是否是前沿：4 邻域中有“已探索自由格”
            is_frontier = False
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H:
                    if global_map[ny, nx] == 0 and explored_map[ny, nx] != -1:
                        is_frontier = True
                        break

            if is_frontier:
                H_f = 1.0  # TODO(full): 根据后续的实际建图算法换成真正的 occupancy 概率熵
                frontier_list.append((x, y, H_f))

    return frontier_list

def predict_coverage_after_moves(env_state, q):
    """
    Predict global cover rate according to new positions

    TODO(full-TRACE-global-coverage):
    - 当前预测仅计算“占据自由栅格”的 coverage ratio \in [0,1]，且使用简单半径+无遮挡模型；
    - 完整版需要：
      1) 使用真实的传感器模型：
           - 考虑 LOS / 遮挡；
           - 使用概率占据映射更新熵 H(x)；
      2) 将标量 coverage 扩展为：
           - 区域化 coverage 向量 z^coverage \in R^R（按子区域划分）；
           - 或完整熵场的某种低维嵌入（如主成分/块平均）；
      3) 使 z^coverage 的定义与理论中 entropy field 的全局量完全对齐，
         以便外层 RL 的 reward 直接使用 z^coverage 或 \Delta z^coverage。
    """
    global_map = env_state.global_map
    explored_map = env_state.explored_map.copy()   # 不要改原图
    H, W = global_map.shape

    # 模拟所有机器人在 q_i 处感知一次
    for rid, q_i in q.items():
        robot = env_state.robots[rid]
        nx, ny = q_i.pos.astype(int)
        r_sense = robot.r_sense

        for dx in range(-r_sense, r_sense + 1):
            for dy in range(-r_sense, r_sense + 1):
                x = nx + dx
                y = ny + dy
                if not (0 <= x < W and 0 <= y < H):
                    continue
                # 半径约束（欧氏距离）
                if dx * dx + dy * dy > r_sense * r_sense:
                    continue
                # 只在可行走区域上揭露
                if global_map[y, x] != 0:
                    continue
                # 标记为已探索（>0 即可）
                explored_map[y, x] = 1

    # 计算预测后的覆盖率
    free_mask = (global_map == 0)
    total_free = int(np.count_nonzero(free_mask))
    if total_free == 0:
        return 0.0

    explored_free = int(np.count_nonzero(free_mask & (explored_map > 0)))
    return explored_free / total_free