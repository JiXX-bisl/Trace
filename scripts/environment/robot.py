"""
robot.py
Created by JiXX at 20251202
- Define uav and ugv node:
    * ugv: ground robot;
    * uav: areaial robot;
- Provide moving and state update interface
"""
import numpy as np


class UAV:
    def __init__(self, id):
        self.type_id = 1
        self.id = id
        self.name = None
        self.pos = np.array([0, 0])
        self.r_safe = 1
        self.r_sense = 6
        self.r_move = 6
        self.local_map = None
        self.target_cell = None
        self.sequence = {}
    
    def get_name(self):
        self.name = f"{self.type_id}_{self.id}"

    def update_pos(self, new_x, new_y):
        # 注意这里的位置在访问栅格的时候需要交换x,y的位置
        self.pos = np.array([new_x, new_y])
    
    def update_target_cell(self, cell_x, cell_y):
        self.target_cell = np.array([cell_x, cell_y], dtype=int)
    

class UGV:
    def __init__(self, id):
        self.type_id = 0
        self.id = id
        self.name = None
        self.pos = np.array([0, 0])
        self.r_safe = 0
        self.r_sense = 3
        self.r_move = 3
        self.local_map = None
        self.target_cell = None
        self.sequence  = {}

    def get_name(self):
        self.name = f"{self.type_id}_{self.id}"

    def update_pos(self, new_x, new_y):
        # 注意这里的位置在访问栅格的时候需要交换x,y的位置
        self.pos = np.array([new_x, new_y])

    def update_target_cell(self, cell_x, cell_y):
        self.target_cell = np.array([cell_x, cell_y], dtype=int)

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

#  这里需要替换为Inner ADMM输出的轨迹
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

def enumerate_candidate_moves(env_state, rid):
    robot = env_state.robots[rid]
    x0, y0 = robot.pos
    r = robot.r_move
    h, w = env_state.global_map.shape

    candidates = []
    for dx in range(-r, r+1):
        for dy in range(-r, r+1):
            nx, ny = x0 + dx, y0 + dy
            if not (0 <= nx < w and 0 <= ny < h): continue
            if max(abs(dx), abs(dy)) > r: continue
            if is_valid_position(env_state.global_map, robot, nx, ny): candidates.append((nx, ny))
    if len(candidates) == 0: candidates.append((x0, y0))
    return candidates
