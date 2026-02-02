"""
visualizer.py
Created by JiXX at 20261203
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


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


class SimpleVisual:
    def __init__(self, robots, dynamic_map, comm_map):
        self.robots = robots
        self.fig, self.axs = plt.subplots(2, 4, figsize=(15, 10))  # 2x4 固定：0,0全局；1,0通信；其余6个机器人
        self.cmap = ListedColormap(['gray', 'white', 'black', "red", "green"])
        # --- Global dynamic map ---
        self.axs[0, 0].set_title("Dynamic Map (Global Exploration)")
        self.visual_dynamic = self.to_visual_dynamic(dynamic_map)
        self.im_dynamic = self.axs[0, 0].imshow(self.visual_dynamic, cmap=self.cmap, vmin=0, vmax=4, origin='lower')
        xs = [r.pos[0] for r in robots]
        ys = [r.pos[1] for r in robots]
        self.dscat = self.axs[0, 0].scatter(xs, ys, c='blue', s=10)

        # --- Local maps (6 robots -> fill (0,1)(0,2)(0,3)(1,1)(1,2)(1,3)) ---
        self.im_locals = []
        self.scat_locals = []
        for i, rbt in enumerate(robots):
            rr = i // 3
            cc = i % 3 + 1
            ax = self.axs[rr, cc]
            robot_type = "UAV" if rbt.type_id == 1 else "UGV"
            ax.set_title(f"{robot_type}_{rbt.id} Local Map")
            ax.set_xticks([]); ax.set_yticks([])
            ax.grid(which='both', color='black', linewidth=0.3)

            vis = self.to_visual_dynamic(rbt.local_map)
            im = ax.imshow(vis, cmap=self.cmap, vmin=0, vmax=4, origin='lower')
            sc = ax.scatter(rbt.pos[0], rbt.pos[1], c='blue', s=50)

            self.im_locals.append(im)
            self.scat_locals.append(sc)

        # --- Comm graph in (1,0) ---
        visualize_communication_graph(comm_map, ax=self.axs[1, 0])

        plt.ion()
        plt.show()

    def update(self, comm_map, dynamic_map, robots):
        # 1) 防呆：避免 shape() 问题
        if dynamic_map is None or not isinstance(dynamic_map, np.ndarray) or dynamic_map.ndim != 2:
            raise ValueError(f"dynamic_map invalid: type={type(dynamic_map)}, shape={np.shape(dynamic_map)}")

        # 2) 更新全局图（不要重复 imshow）
        self.visual_dynamic = self.to_visual_dynamic(dynamic_map)
        self.im_dynamic.set_data(self.visual_dynamic)

        xs = [r.pos[0] for r in robots]
        ys = [r.pos[1] for r in robots]
        self.dscat.set_offsets(np.c_[xs, ys])

        # 3) 更新每个机器人 local map & 位置（不要重复 imshow / scatter）
        for i, rbt in enumerate(robots):
            self.im_locals[i].set_data(self.to_visual_dynamic(rbt.local_map))
            self.scat_locals[i].set_offsets([[rbt.pos[0], rbt.pos[1]]])

        # 4) 更新通信图（这个可以 clear+重画）
        visualize_communication_graph(comm_map, ax=self.axs[1, 0])

        self.fig.canvas.draw_idle()
        plt.pause(0.4)

    def to_visual_dynamic(self, visual_map):
        # visual_map 必须是 2D ndarray
        visual = np.zeros_like(visual_map, dtype=int)
        visual[visual_map == -1] = 0
        visual[visual_map == 0]  = 1
        visual[visual_map == 1]  = 2
        visual[visual_map == 2]  = 3
        visual[visual_map == 3]  = 4
        return visual
