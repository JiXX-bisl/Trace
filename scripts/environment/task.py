"""
task.py
Created by JiXX at 20251203
Describe simple task system
"""
from scripts.core.data import Task

def spawn_random_task(global_map, explored_map, tasks:list, current_step, rng, min_lt=15, max_lt=20):
    """
    在可通行栅格上随机生成一个 COI 任务：
    - 生成时间 = current_step
    - 有效时间 = [start_step, deadline_step]
    - 在 global_map 上把该格设为 3（COI）
    """
    h, w = global_map.shape
    for _ in range(1000):
        x = int(rng.integers(w))
        y = int(rng.integers(h))
        # 只在自由格上生成（global_map==0），避免障碍物上
        if global_map[y, x] != 0 or explored_map[y, x] == -1:
            continue
        # 避免在已有任务位置重复生成
        if any(t.x == x and t.y == y and t.status not in ("failed", "completed")
               for t in tasks):
            continue

        lifetime = int(rng.integers(min_lt, max_lt + 1))
        task_id = len(tasks)
        tasks.append(Task(
            id=task_id,
            x=x,
            y=y,
            start_step=current_step,
            ddl_step=current_step + lifetime, 
        ))
        global_map[y, x] = 2  # 标记为 COI
        # 也可以在 dynamic_map 可视化中用特殊颜色区分
        # print(f"[step {current_step}] spawn task {task_id} at ({x},{y}), deadline={current_step + lifetime}")
        return