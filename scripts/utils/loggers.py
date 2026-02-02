"""
logger.py
Created by JiXX at 20251203
Record process parameters
"""
import os
import csv
import json
import numpy as np

class ExplorationLogger:
    def __init__(self, robots, log_dir="logs", run_name="run", strong_th=3, save_maps_every=0):
        """
        strong_th: 你定义的强通信阈值（比如 signal > 3）
        save_maps_every: >0 时每隔多少步保存一次 .npy 地图快照（0 表示不保存）
        """
        self.robots = robots
        self.n = len(robots)
        self.strong_th = strong_th
        self.save_maps_every = save_maps_every

        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, f"{run_name}.csv")
        self.jsonl_path = os.path.join(log_dir, f"{run_name}.jsonl")
        self.log_dir = log_dir

        # CSV header
        header = ["step",
                  "explore_rate_all", "explore_rate_free",
                  "known_cells_all", "total_cells_all",
                  "known_cells_free", "total_cells_free"]

        # robot positions
        for i, r in enumerate(self.robots):
            header += [f"r{i}_type", f"r{i}_id", f"r{i}_x", f"r{i}_y"]

        # comm signals (i -> j)
        for i in range(self.n):
            for j in range(self.n):
                if i == j: 
                    continue
                header += [f"sig_{i}_{j}", f"conn_{i}_{j}", f"strong_{i}_{j}"]

        self._csv_f = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._csv_w = csv.writer(self._csv_f)
        self._csv_w.writerow(header)
        self._csv_f.flush()

        self._jsonl_f = open(self.jsonl_path, "w", encoding="utf-8")

    def close(self):
        self._csv_f.close()
        self._jsonl_f.close()

    @staticmethod
    def _exploration_rates(global_map, dynamic_map):
        # 口径A：全部格子（含障碍）
        known_all = int(np.count_nonzero(dynamic_map != -1))
        total_all = int(dynamic_map.size)
        rate_all = known_all / total_all if total_all else 0.0

        # 口径B：只统计可通行/非障碍格子（更常用）
        free_mask = (global_map != 1)
        known_free = int(np.count_nonzero((dynamic_map != -1) & free_mask))
        total_free = int(np.count_nonzero(free_mask))
        rate_free = known_free / total_free if total_free else 0.0

        return (rate_all, rate_free,
                known_all, total_all,
                known_free, total_free)

    def log_step(self, step, global_map, dynamic_map, comm_map):
        # 1) exploration
        (rate_all, rate_free,
         known_all, total_all,
         known_free, total_free) = self._exploration_rates(global_map, dynamic_map)

        # 2) positions
        row = [step, rate_all, rate_free, known_all, total_all, known_free, total_free]
        robots_payload = []
        for i, r in enumerate(self.robots):
            x, y = int(r.pos[0]), int(r.pos[1])
            row += [int(r.type_id), int(r.id), x, y]
            robots_payload.append({"i": i, "type_id": int(r.type_id), "id": int(r.id), "x": x, "y": y})

        # 3) comm (signal + connected + strong)
        comm_payload = []
        node_names = [f"{r.type_id}_{r.id}" for r in self.robots]
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                u, v = node_names[i], node_names[j]
                data = comm_map.get_edge_data(u, v)
                if data is None:
                    sig = np.nan
                    conn = 0
                    strong = 0
                else:
                    sig = float(data.get("signal", np.nan))
                    conn = 1
                    strong = 1 if (sig > self.strong_th) else 0

                row += [sig, conn, strong]
                comm_payload.append({"from": i, "to": j, "signal": None if np.isnan(sig) else sig,
                                     "connected": bool(conn), "strong": bool(strong)})

        # write CSV
        self._csv_w.writerow(row)
        self._csv_f.flush()

        # write JSONL (结构化更好读)
        payload = {
            "step": int(step),
            "explore_rate_all": rate_all,
            "explore_rate_free": rate_free,
            "robots": robots_payload,
            "comm": comm_payload
        }
        self._jsonl_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._jsonl_f.flush()

        # optional map snapshots
        if self.save_maps_every and (step % self.save_maps_every == 0):
            np.save(os.path.join(self.log_dir, f"dynamic_step_{step:04d}.npy"), dynamic_map)
            for i, r in enumerate(self.robots):
                np.save(os.path.join(self.log_dir, f"local_r{i}_step_{step:04d}.npy"), r.local_map)
