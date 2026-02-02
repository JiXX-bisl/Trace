"""
communication.py
Created by JiXX at 20251202
Provide simple communication simulation according to number of free cells and obstacle cells
"""
from dataclasses import dataclass
import networkx as nx
import numpy as np
import math; calculate_dist = lambda x1,y1,x2,y2: math.sqrt((x2-x1)**2 + (y2-y1)**2)

from scripts.core.data import LinkState
from scripts.utils.helper import bresenham_line, shortest_path_edge_gt


UAV_RANGE=25
UGV_RANGE=25
BASE=80
c_5=0.2;c_15=6.0;c_25=20.0
d_5=250; d_15=80;d_25=25
p_5=0.6;p_15=0.1;p_25=0.01


def update_comm_map(global_map, comm_map: nx.DiGraph, robots):
    for i, robot1 in enumerate(robots):
        for j, robot2 in enumerate(robots):
            if i == j: continue
            line = bresenham_line(robot1.pos[0], robot1.pos[1], 
                                  robot2.pos[0], robot2.pos[1])
            if len(line) <= 2:
                signal = 20
            else:
                free_cells = 0
                obs_cells = 0
                for (x, y) in line[1:-1]:
                    if global_map[y, x] == 1:
                        obs_cells += 1
                    else:
                        free_cells += 1
                prop = free_cells * 1 + obs_cells * 5
                if robot1.type_id == 0:
                    signal = UGV_RANGE - prop
                else:
                    signal = UAV_RANGE - prop
            signal = max(0, signal)
            # if signal > 5:
            comm_map.add_edge(f"{robot1.name}", f"{robot2.name}", signal=signal)
    for i, robot in enumerate(robots):
        line = bresenham_line(comm_map.nodes["base"]["pos"][0], comm_map.nodes["base"]["pos"][1], 
                              robot.pos[0], robot.pos[1])
        if len(line) <= 2:
            signal = 25
        else:
            free_cells = 0
            obs_cells = 0
            for (x, y) in line[1:-1]:
                if global_map[y, x] == 1:
                    obs_cells += 1
                else:
                    free_cells += 1
            prop = free_cells * 1 + obs_cells * 5
            signal = BASE - prop
        signal = max(0, signal)
        # if signal > 5:
        comm_map.add_edge("base", f"{robot.name}", signal=signal)
        comm_map.add_edge(f"{robot.name}", "base", signal=signal)


def get_link_state(comm_map: nx.DiGraph):
    link_state = []
    for u, _ in comm_map.nodes(data=True):
        for v, _ in comm_map.nodes(data=True):
            if u == v: continue
            if not comm_map.has_edge(u, v): continue
            e_attr = comm_map.edges[u, v]
            signal = e_attr.get("signal", None)
            # 0-5: 0 cell 5-15: 100, >15: 1000
            if signal < 3: 
                signal = max(signal, 0)
                cap = 0.0; delay = np.infty; plr = 1.0
            elif signal> 15:
                x = (signal - 15) / 10.0
                cap = c_15 + (c_25 - c_15) * x
                delay = d_15 * (d_25 / d_15) * x
                plr = p_15 * (p_25 / p_15) ** x
            else:
                x = (signal - 5) / 10.0
                cap = c_5 + (c_15 - c_5) * x
                delay = d_5 * (d_15 / d_5) * x
                plr = p_5 * (p_15 / p_5) ** x
                # d_5=250; d_15=80;d_25=25
            link_state.append(LinkState(
                tx=u,
                rx=v,
                rssi=signal, 
                capacity=cap, 
                delay=delay,
                plr=plr
            ))
    return link_state

def get_main_route(comm_map: nx.DiGraph, robots, task):
    """
    Get route for each task according to the robot positions
    """
    if task.status == "completed" or task.status == "failed": return []
    # 计算该task与所有机器人节点位置关系
    candidate_robots = {}
    for rid, robot in enumerate(robots):
        dist = calculate_dist(task.x, task.y, robot.pos[0], robot.pos[1])
        if dist <= robot.r_sense:
            candidate_robots[rid] = dist
    if len(candidate_robots) == 0:
        task.isOnline = False
        return []
    for rid in candidate_robots:
        paths = shortest_path_edge_gt(comm_map, f"{robots[rid].type_id}_{rid}", "base", threshold=3.0)
        if len(paths) != 0:
            task.isOnline = True
            return paths
        else:
            task.isOnline = False
            return []

def estimate_link_state(env_state, pos_cur, pos_nb):
    line = bresenham_line(pos_cur[0], pos_cur[1], 
                            pos_nb[0], pos_nb[1])
    if len(line) <= 2:
        signal = 20
    else:
        free_cells = 0
        obs_cells = 0
        for (x, y) in line[1:-1]:
            if env_state.global_map[int(y), int(x)] == 1:
                obs_cells += 1
            else:
                free_cells += 1
        prop = free_cells * 1 + obs_cells * 5
        signal = 25 - prop
    signal = max(0, signal)
    return signal
