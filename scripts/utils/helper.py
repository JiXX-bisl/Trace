"""
helper.py
Created by JiXX at 20251202
Provides helper functions
- has_line_of_sight: whether LoS between input nodes
- bresenham_line: return grid pos between two input nodes
"""
import numpy as np
import math
import networkx as nx
import heapq

from scripts.core.data import EnvState


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


def compute_current_coverage_ratio(env_state: EnvState) -> float:
    """
    根据当前 explored_map / coverage_state 计算全局覆盖率。

    最小实现：
    - 以 global_map==0 的自由栅格为全集；
    - explored_map > 0 视为已探索；
    - 覆盖率 = 已探索自由格子数 / 自由格子总数。

    TODO(full):
    - 换成信息熵/信息增益定义；
    - 对 COI 区域加权；
    - 按子区域分块，扩展为 coverage 向量 z^coverage ∈ R^R。
    """
    global_map = env_state.global_map
    explored_map = env_state.explored_map  # 需要你在 EnvState 里保证有这个字段

    free_mask = (global_map == 0)
    total_free = int(np.count_nonzero(free_mask))
    if total_free == 0:
        return 0.0

    explored_mask = (explored_map >= 0)
    explored_free = int(np.count_nonzero(free_mask & explored_mask))
    return explored_free / total_free

def entropy_potential_at_pos(
    nx: int,
    ny: int,
    frontier_entropy,
    max_radius: int = 10,
    sigma: float = 5.0,
) -> float:
    """
    计算候选位置 (nx, ny) 的熵势 Phi(p)。

    最小实现：
    - 对所有前沿单元 f=(fx,fy,Hf)：
        若 d(p,f)^2 <= max_radius^2:
            贡献 Hf * exp(-d^2/(2σ^2))
    - 距离采用曼哈顿距离或欧氏距离（这里用欧氏的平方）

    参数:
    - frontier_entropy: [(fx, fy, Hf), ...]
    - max_radius: 截断半径，避免全图 O(N^2) 计算
    - sigma: 熵场扩散尺度（越大越“平”）

    返回:
    - Phi(p): 熵势值，越大表示位置越“信息丰富”。
    
    TODO(full-TRACE-entropy-interaction):
    - 当前使用简单的高斯核 Φ(p) = Σ H_f exp(-d^2/(2σ^2))，忽略 LOS/遮挡；
    - 完整版可扩展为：
      1) 仅对 LOS 可见 or 感知模型可达的 frontier cell 计入熵势；
      2) 使用 graph-based 距离（比如在可行空间图上的最短路长度）代替简单欧氏距离；
      3) 区分多个 main 之间的竞争/协作，在 Φ(p) 中引入“多机器人排斥项”
         （避免所有 main 跑去同一个 frontier）。
    """
    if not frontier_entropy:
        return 0.0

    total = 0.0
    max_r2 = max_radius * max_radius

    for fx, fy, Hf in frontier_entropy:
        dx = fx - nx
        dy = fy - ny
        d2 = dx * dx + dy * dy
        if d2 > max_r2:
            continue
        # 欧氏距离的高斯核
        # 注意：sigma 太小 ⇒ 熵场非常局部；太大 ⇒ 变成平摊常数
        w = math.exp(- d2 / (2.0 * sigma * sigma))
        total += Hf * w

    return total

def update_repulsion_grad(aux, q_ref):
    """
    根据当前参考位置 q_ref（通常是上一轮 q_i）计算每个机器人的 repulsion 梯度 g_i，
    并写回 aux["repulsion_grad"]。

    思路：
    - 定义一个随距离 d 衰减的势函数 φ(d) = exp(-d / sigma)
    - 取其导数 φ'(d) = -(1/sigma) * exp(-d/sigma)
    - 对每个 i，累加所有 j≠i 的贡献：
        g_i = Σ_j φ'(d_ij) * (q_i - q_j) / d_ij
      这是 φ 对 q_i 的梯度 ∇_{q_i} φ 的近似。
    - 在 cost 里用线性项 g_i^T (q_i - q_i_prev) 近似 repulsion 带来的变化。
    """
    robot_ids = aux["robot_ids"]
    sigma = float(aux.get("repulsion_sigma", 2.0))

    repulsion_grad = {}

    for rid_i in robot_ids:
        qi = q_ref[rid_i]
        g_i = np.zeros(2, dtype=float)

        for rid_j in robot_ids:
            if rid_j == rid_i:
                continue
            qj = q_ref[rid_j]
            diff = qi.pos - qj.pos
            d = float(np.linalg.norm(diff) + 1e-6)
            # 单位方向向量 v_ij: j -> i
            v_ij = diff / d
            # φ(d) = exp(-d/sigma) -> φ'(d) = -(1/sigma)*exp(-d/sigma)
            phi_prime = -(1.0 / sigma) * math.exp(-d / sigma)
            g_i += phi_prime * v_ij

        repulsion_grad[rid_i] = g_i

    aux["repulsion_grad"] = repulsion_grad

def role_priority(role):
    """
    冲突消解时用的角色优先级：
    - main（主轴） > relay > free
    这里兼容两种写法：
    - 整数角色 id（当前实现中：0-main，1-free，2-relay 的风格）
    - 字符串角色："main" / "relay" / "free"
    """
    # 字符串形式
    if isinstance(role, str):
        mapping = {"main": 3, "relay": 2, "free": 1}
        return mapping.get(role, 1)

    # 整数形式（根据你当前 cost 里的约定：role == 0 为主轴探索）
    if role == 0:   # main
        return 3
    if role == 2:   # relay（假定 2 是 relay）
        return 2
    # 其他都当 free
    return 1

def shortest_path_edge_gt(G: nx.DiGraph, src, base, weight_key="signal", threshold=3.0):
    if src not in G or base not in G:
        return []

    dist = {src: 0.0}
    prev = {}
    pq = [(0.0, src)]

    best_cost = math.inf
    best_prev = None  # 用于回溯 base 的父节点
    best_edge_w = None

    while pq:
        d, u = heapq.heappop(pq)
        if d != dist.get(u, math.inf):
            continue

        # 早停判定：堆顶已经不可能改进 best
        if d >= best_cost:
            break

        # “与 base 建立连接判定”（u -> base 的直连边是否可用）
        if G.has_edge(u, base):
            w = G[u][base].get(weight_key, None)
            if w is not None and w > threshold:
                cand = d + float(w)
                if cand < best_cost:
                    best_cost = cand
                    best_prev = u
                    best_edge_w = w

        # 正常松弛：只走权重>threshold 的边
        for v, attr in G.succ[u].items():
            w = attr.get(weight_key, None)
            if w is None or w < threshold:
                continue
            if w < 0:
                raise ValueError("Dijkstra 需要非负权重")
            nd = d + float(w)
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if best_cost == math.inf:
        return []

    # 回溯：src -> ... -> best_prev -> base
    path = [best_prev]
    while path[-1] != src:
        path.append(prev[path[-1]])
    path.reverse()
    path.append(base)
    return path

def shortest_multihop_route_grid(
    global_map,
    tx,   # (x, y)
    rx,   # (x, y)
    threshold,
    RANGE,
):
    """
    在 grid 上找 tx->rx 的最短多跳路径（默认 hop 最少）；
    约束：每一跳两端格子间的 signal > threshold。
    返回：[(x,y), (x,y), ...]，找不到返回 []。
    """
    H, W = global_map.shape
    (sx, sy), (gx, gy) = tx, rx

    def in_bounds(x, y):
        return 0 <= x < W and 0 <= y < H

    if not in_bounds(sx, sy) or not in_bounds(gx, gy):
        return []
    if global_map[sy, sx] == 1 or global_map[gy, gx] == 1:
        return []

    # 可用边距离上界（Chebyshev）
    Dmax = int(RANGE - threshold)
    if Dmax <= 0:
        return []

    # -------- signal 计算（带缓存 + 早停）--------
    sig_cache = {}

    # 预先算一下 prop 允许的上限：RANGE - prop > threshold -> prop < RANGE - threshold
    prop_limit = RANGE - threshold

    def signal_ok(a, b):
        # a,b: (x,y)
        key = (a[0], a[1], b[0], b[1])
        if key in sig_cache:
            return sig_cache[key]

        x0, y0 = a
        x1, y1 = b
        cheb = max(abs(x1 - x0), abs(y1 - y0))
        # 必要条件：cheb <= Dmax，否则一定不可能
        if cheb > Dmax:
            sig_cache[key] = False
            return False

        line = bresenham_line(x0, y0, x1, y1)
        if len(line) <= 2:
            ok = (20 > threshold) if RANGE >= 20 else (RANGE > threshold)
            sig_cache[key] = ok
            return ok

        # prop = free*1 + obs*5
        prop = 0.0
        # 中间点
        for (x, y) in line[1:-1]:
            if global_map[y, x] == 1:
                prop += 5.0
            else:
                prop += 1.0
            # 早停：prop 已经达到/超过 prop_limit，则 signal <= threshold
            if prop >= prop_limit:
                sig_cache[key] = False
                return False

        ok = (RANGE - prop) > threshold
        sig_cache[key] = ok
        return ok

    # -------- A*（代价=hop数）--------
    def heuristic(x, y):
        # 每跳最多推进 Dmax 的 Chebyshev 距离，因此下界为 ceil(cheb/Dmax)
        cheb = max(abs(gx - x), abs(gy - y))
        return (cheb + Dmax - 1) // Dmax

    start = (sx, sy)
    goal = (gx, gy)

    g_cost = {start: 0}
    prev = {}  # node -> parent

    open_heap = []
    heapq.heappush(open_heap, (heuristic(sx, sy), 0, start))

    while open_heap:
        f, g, u = heapq.heappop(open_heap)
        if g != g_cost.get(u, None):
            continue

        if u == goal:
            # 回溯路径
            path = [goal]
            while path[-1] != start:
                path.append(prev[path[-1]])
            path.reverse()
            return path

        ux, uy = u

        # 枚举候选下一跳：限制在 Dmax 方窗内
        for dx in range(-Dmax, Dmax + 1):
            nx = ux + dx
            if nx < 0 or nx >= W:
                continue
            for dy in range(-Dmax, Dmax + 1):
                if dx == 0 and dy == 0:
                    continue
                ny = uy + dy
                if ny < 0 or ny >= H:
                    continue

                v = (nx, ny)
                if global_map[ny, nx] == 1:
                    continue

                # 再次用 Chebyshev 限制（可少算一点）
                if max(abs(dx), abs(dy)) > Dmax:
                    continue

                if not signal_ok(u, v):
                    continue

                ng = g + 1
                if ng < g_cost.get(v, 10**18):
                    g_cost[v] = ng
                    prev[v] = u
                    heapq.heappush(open_heap, (ng + heuristic(nx, ny), ng, v))

    return []
