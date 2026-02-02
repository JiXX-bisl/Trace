import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class UAV:
    def __init__(self, id):
        self.type_id = 1
        self.id = id
        self.pos = np.array([0, 0])
        self.r_safe = 1
        self.r_sense = 6
        self.r_move = 5
        self.local_map = None

    def update_pos(self, new_x, new_y):
        # 注意这里的位置在访问栅格的时候需要交换x,y的位置
        self.pos = np.array([new_x, new_y])


class UGV:
    def __init__(self, id):
        self.type_id = 0
        self.id = id
        self.pos = np.array([0, 0])
        self.r_safe = 0
        self.r_sense = 3
        self.r_move = 2
        self.local_map = None

    def update_pos(self, new_x, new_y):
        # 注意这里的位置在访问栅格的时候需要交换x,y的位置
        self.pos = np.array([new_x, new_y])

def is_valid_position(global_map, robot, x, y):
    """
    检查机器人是否处于合法的位置
    - 在地图范围内
    - 不在障碍物上
    - 满足安全半径约束
    """
    h, w = global_map.shape
    if not (0 <= x < w and 0 <= y < h): return False
    if global_map[y, x] == 1: return False

    r_safe = robot.r_safe
    if r_safe > 0:
        for dx in range(-r_safe, r_safe + 1):
            for dy in range(-r_safe, r_safe + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    if global_map[ny, nx] == 1 or global_map[ny, nx] == -1:
                        return False
    return True

def bresenham_line(x0, y0, x1, y1):
    """Bresenham 算法生成从 (x0, y0) 到 (x1, y1) 的离散栅格坐标列表。"""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x1 >= x0 else -1
    sy = 1 if y1 >= y0 else -1

    if dy <= dx:
        err = dx // 2
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
        points.append((x, y))
    else:
        err = dy // 2
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
        points.append((x, y))
    return points

def has_line_of_sight(global_map, x0, y0, x1, y1):
    """
    判断机器人 (x0,y0) 与目标格子 (x1,y1) 之间是否有视线：
    - 使用 Bresenham 线
    - 允许看到目标格子本身为障碍物
    - 若中途经过的任何格子为障碍物，则视线被阻挡
    """
    line = bresenham_line(x0, y0, x1, y1)
    if len(line) <= 2:
        return True
    for (x, y) in line[1:-1]:
        if global_map[y, x] == 1:
            return False
    return True


def reveal_with_robot(global_map, dynamic_map, robot):
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

def update_local_maps_based_on_communication(global_map, comm_map: nx.DiGraph, robots):
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
    
    # 更新完所有机器人后，可以根据需求进一步处理 (如显示地图等)
    print("局部地图更新完成")


def random_move_robot(global_map, robot, rng):
    """
    让机器人在其允许的移动半径内随机移动一步：
    - 使用 Chebyshev 距离 <= r_move
    - 不能落在障碍物上
    - 满足安全半径约束
    - 允许停在原地（视作不动）
    """
    h, w = global_map.shape
    x0, y0 = robot.pos
    r_move = robot.r_move

    candidates = []
    for dx in range(-r_move, r_move + 1):
        for dy in range(-r_move, r_move + 1):
            nx, ny = x0 + dx, y0 + dy
            if not (0 <= nx < w and 0 <= ny < h): continue
            if max(abs(dx), abs(dy)) > r_move: continue
            if is_valid_position(global_map, robot, nx, ny): candidates.append((nx, ny))
    if not candidates: return

    new_x, new_y = candidates[rng.integers(len(candidates))]
    robot.update_pos(new_x, new_y)

def to_visual_map(dynamic_map):
    visual = np.zeros_like(dynamic_map, dtype=int)
    visual[dynamic_map==-1] = 0
    visual[dynamic_map==0] = 1
    visual[dynamic_map==1] = 2
    return visual

def update_comm_map(global_map, comm_map: nx.DiGraph, robots):
    for i, robot1 in enumerate(robots):
        for j, robot2 in enumerate(robots):
            if i == j: continue
            line = bresenham_line(robot1.pos[0], robot1.pos[1], 
                                  robot2.pos[0], robot2.pos[1])
            if len(line) <= 2:
                return True
            free_cells = 0
            obs_cells = 0
            for (x, y) in line[1:-1]:
                if global_map[y, x] == 1:
                    obs_cells += 1
                else:
                    free_cells += 1
            prop = free_cells * 1 + obs_cells * 5
            if robot1.type_id == 0:
                signal = 20 - prop
            else:
                signal = 25 - prop
            if signal > -100:
                comm_map.add_edge(f"{robot1.type_id}_{robot1.id}", f"{robot2.type_id}_{robot2.id}", signal=signal)

def init_robots(global_map, rng):
    """
    初始化 3 台 UGV 和 3 架 UAV，随机放置在地图上的合法位置。
    """
    robots = []
    h, w = global_map.shape

    def sample_position_for(robot):
        for _ in range(1000):
            x = int(rng.integers(w))
            y = int(rng.integers(h))
            if x > 10 or y > 10: continue  # 保证机器人从初始节点出发
            if is_valid_position(global_map, robot, x, y):
                # 避免和已有机器人重合
                if all(not (r.pos[0] == x and r.pos[1] == y) for r in robots):
                    return x, y
        raise RuntimeError("无法为机器人找到合法初始位置，请检查地图是否过于拥挤。")

    # 先创建机器人对象
    for i in range(3):
        robots.append(UGV(id=i))
    for i in range(3):
        robots.append(UAV(id=i))

    # 为每个机器人随机分配初始位置
    for robot in robots:
        x, y = sample_position_for(robot)
        robot.update_pos(x, y)
        robot.local_map = -np.ones_like(global_map)
    

    return robots

def visualize_communication_graph(G: nx.DiGraph, ax=None):
    """
    在给定的 ax 上可视化通信图，并显示每条边的正/反向信号强度
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    ax.clear()
    ax.set_title("Communication Graph")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    ax.axis("off")

    pos = nx.get_node_attributes(G, 'pos')
    if not pos:  # 如果没有pos属性，就用layout兜底
        pos = nx.spring_layout(G, seed=0)

    edge_weights = nx.get_edge_attributes(G, 'signal')

    # 可选：去重（避免(u,v)和(v,u)都写一遍双向标签导致爆炸）
    edge_labels = {}
    seen = set()
    for (u, v) in G.edges:
        key = tuple(sorted((u, v)))
        if key in seen:
            continue
        seen.add(key)
        fwd = edge_weights.get((u, v), 0)
        rev = edge_weights.get((v, u), 0)
        # 注意：标签挂在(u,v)这条边上（如果不存在(u,v)但存在(v,u)，可再做个判断）
        edge_labels[(u, v)] = f"{u}->{v}:{fwd}\n{v}->{u}:{rev}"

    nx.draw(
        G, pos, ax=ax,
        with_labels=True, node_color='lightblue',
        node_size=800, font_size=12, font_weight='bold',
        edge_color='gray', arrows=True
    )
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels,
        ax=ax, font_size=9, font_color='red'
    )

def simulate_exploration(global_map, num_steps=20, seed=42):
    rng = np.random.default_rng(seed)
    h, w = global_map.shape
    dynamic_map = -np.ones_like(global_map)  # 未知区域
    robots = init_robots(global_map, rng)
    cmap = ListedColormap(['gray', 'white', 'black'])

    # 创建7个子图，1个显示dynamic_map，剩余6个显示每个机器人的local_map
    fig, axs = plt.subplots(2, 4, figsize=(15, 10))  # 2行4列，共7个图
    axs[0, 0].set_title("Dynamic Map (Global Exploration)")
    # 在dynamic_map位置显示所有机器人动态演化
    visual_dynamic = to_visual_map(dynamic_map=dynamic_map)
    axs[0, 0].imshow(visual_dynamic, cmap, vmin=0, vmax=2, origin='lower')
    xs = [robot.pos[0] for robot in robots]
    ys = [robot.pos[1] for robot in robots]
    dscat = axs[0, 0].scatter(xs, ys, c='blue', s=50)    # 初始化机器人位置的散点图

    axs[1, 0].set_title("Communication Graph")
    axs[1, 0].set_xticks([]); axs[1, 0].set_yticks([])
    axs[1, 0].axis("off")

    scat = []
    visual_locals = []
    for i, robot in enumerate(robots):
        visual_locals.append(to_visual_map(dynamic_map=robot.local_map))
        axs[i//3, i%3+1].imshow(visual_locals[i], cmap, vmin=0, vmax=2, origin='lower')  # 显示每个机器人local map
        axs[i//3, i%3+1].set_title(f"Robot {i+1} Local Map")
        axs[i//3, i%3+1].set_xticks([])
        axs[i//3, i%3+1].set_yticks([])
        axs[i//3, i%3+1].grid(which='both', color='lightgray', linewidth=0.3)

        scat.append(axs[i//3, i%3+1].scatter(robot.pos[0], robot.pos[1], c='blue', s=50))

    plt.ion()
    plt.show()

    comm_map = nx.DiGraph()
    for robot in robots:
        comm_map.add_node(f"{robot.type_id}_{robot.id}", pos=robot.pos)

    for robot in robots:
        reveal_with_robot(global_map, dynamic_map, robot)
    update_comm_map(global_map, comm_map, robots)
    update_local_maps_based_on_communication(global_map, comm_map, robots)
    visualize_communication_graph(comm_map, ax=axs[1, 0])

    for step in range(num_steps + 1):
        # 更新dynamic_map
        visual_dynamic = to_visual_map(dynamic_map)
        axs[0, 0].imshow(visual_dynamic, cmap, vmin=0, vmax=2, origin='lower')
        xs = [robot.pos[0] for robot in robots]
        ys = [robot.pos[1] for robot in robots]
        dscat.set_offsets(np.c_[xs, ys])
        # 更新每个机器人的local map
        for i, robot in enumerate(robots):
            visual_locals[i] = to_visual_map(robot.local_map)
            axs[i//3, i%3+1].imshow(visual_locals[i], cmap, vmin=0, vmax=2, origin='lower')
            # 更新机器人位置
            xs = robot.pos[0]
            ys = robot.pos[1]
            scat[i].set_offsets(np.c_[xs, ys])
        visualize_communication_graph(comm_map, ax=axs[1, 0])
        plt.pause(0.4)

        if step == num_steps:
            break

        for robot in robots:
            random_move_robot(global_map, robot, rng)  # 这部分可以替换为C-ADMM决策输出
        update_comm_map(global_map, comm_map, robots)

        for robot in robots:
            reveal_with_robot(global_map, dynamic_map, robot)
        update_local_maps_based_on_communication(global_map, comm_map, robots)

    plt.ioff()
    plt.show()
    
    # visualize_communication_graph(comm_map)



def main():
    global_map = np.load('D:\研究生\TJU\工作情况\9 大论文\实验部分\grid地图最小验证\scripts\environment\gridMap80.npy')
    simulate_exploration(global_map=global_map)

if __name__ == "__main__":
    main()


