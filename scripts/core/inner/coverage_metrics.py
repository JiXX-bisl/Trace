"""
scripts.core.inner.coverage_metrics

Coverage-group preference (y_hat) and coupled sigma construction.

This module is intentionally **pure** and can be unit-tested independently.
It provides:

  * ``coverage_group_scores``: construct raw (unnormalized) scores over G
    coverage groups using local information.
  * ``normalize_to_simplex_nonneg``: produce a nonnegative vector on the simplex
    (sum=1) robustly.
  * ``sigma_target_from_y_hat`` and ``sigma_lower_bound_from_y_hat``: monotone
    couplings used by q-step (preference) and z-step (feasible set).

If later the notion of "coverage groups" changes (e.g., partitions instead of
task.cluster_id), only ``coverage_group_scores`` needs replacement.
"""
from __future__ import annotations

import numpy as np

def _value_at_entropy(frontier_entropy: np.ndarray, xyz: np.ndarray) -> float:
    ent = np.asarray(frontier_entropy, dtype=np.float32)
    if ent.ndim == 3:
        D, H, W = int(ent.shape[0]), int(ent.shape[1]), int(ent.shape[2])
        x = int(np.round(float(xyz[0])))
        y = int(np.round(float(xyz[1])))
        z = int(np.round(float(xyz[2])))
        if (x < 0) or (x >= W) or (y < 0) or (y >= H) or (z < 0) or (z >= D):
            return 0.0
        v = float(ent[z, y, x])
        if not np.isfinite(v):
            return 0.0
        return v
    elif ent.ndim == 2:
        H, W = int(ent.shape[0]), int(ent.shape[1])
        x = int(np.round(float(xyz[0])))
        y = int(np.round(float(xyz[1])))
        if (x < 0) or (x >= W) or (y < 0) or (y >= H):
            return 0.0
        v = float(ent[y, x])
        if not np.isfinite(v):
            return 0.0
        return v
    else:
        return 0.0

def coverage_group_scores(problem: object, rid: int, pos_xy: np.ndarray) -> np.ndarray:
    """
    Construct raw coverage-group scores (G,) for robot ``rid`` at ``pos_xy``.

    Priority order:
      1) Use TaskSnapshot.cluster_id to aggregate tasks into G groups.
      2) If no tasks or all scores are zero, fall back to directional frontier_entropy stats.
      3) If still zero, return all-ones (uniform after normalization).
    """
    _ = rid
    coord_dim = problem.coord_dim
    G = int(getattr(problem, "G", 0) or 0)
    if G <= 0:
        return np.zeros((0,), dtype=np.float32)
    
    params = getattr(problem, "params", None)
    ent = np.asarray(getattr(problem, "frontier_entropy", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
    
    try:
        p = np.asarray(pos_xy, dtype=np.float32).reshape(coord_dim)
    except Exception:
        p = np.zeros((coord_dim,), dtype=np.float32)
    scores = np.zeros((G,), dtype=np.float32)

    # --- task-based grouping ---
    task = getattr(problem, "task", None)
    if task is not None:
        tp = np.asarray(getattr(task, "task_pos", np.zeros((0, coord_dim), dtype=np.float32)), dtype=np.float32).reshape(-1, coord_dim)
        cid = np.asarray(getattr(task, "cluster_id", np.zeros((0,), dtype=np.int32)), dtype=np.int32).reshape(-1)
        T = int(min(tp.shape[0], cid.size))
        if T > 0:
            tp = tp[:T]
            cid = cid[:T]
            dist_scale = float(getattr(params, "coverage_dist_scale", 5.0))
            if not np.isfinite(dist_scale) or dist_scale <= 0:
                dist_scale = 5.0
            for j in range(T):
                g = int(cid[j])
                if g < 0 or g >= G:
                    continue
                e = _value_at_entropy(ent, tp[j])
                d = float(np.linalg.norm(tp[j] - p))
                w = float(np.exp(-d / dist_scale))
                scores[g] += float(e) * w
    
    # --- entropy-based fallback if no tasks or all zero ---
    if not np.any(scores > 0.0):
        if ent.ndim == 2 and ent.size > 0:
            radius = int(getattr(params, "coverage_entropy_radius", 10))
            if radius <= 0:
                radius = 10
            dist_scale = float(getattr(params, "coverage_dist_scale", 5.0))
            if not np.isfinite(dist_scale) or dist_scale <= 0:
                dist_scale = 5.0
            H, W = int(ent.shape[0]), int(ent.shape[1])
            cx = int(np.round(float(p[0])))
            cy = int(np.round(float(p[1])))
            x0, x1 = max(0, cx - radius), min(W - 1, cx + radius)
            y0, y1 = max(0, cy - radius), min(H - 1, cy + radius)
            if (x1 >= x0) and (y1 >= y0):
                for yy in range(y0, y1 + 1):
                    for xx in range(x0, x1 + 1):
                        v = float(ent[yy, xx])
                        if not np.isfinite(v) or v <= 0.0:
                            continue
                        dx = float(xx - cx)
                        dy = float(yy - cy)
                        dist = float(np.hypot(dx, dy))
                        if dist <= 1e-6:
                            continue
                        ang = float(np.arctan2(dy, dx))
                        g = int(np.floor(((ang + np.pi) / (2.0 * np.pi)) * G))
                        g = max(0, min(G - 1, g))
                        scores[g] += v * float(np.exp(-dist / dist_scale))

    if not np.any(scores > 0.0):
        return np.ones((G,), dtype=np.float32)
    return scores.astype(np.float32, copy=False)

def normalize_to_simplex_nonneg(vec: np.ndarray) -> np.ndarray:
    """
    Clip negatives to 0 then normalize to sum = 1
    """
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    G = int(v.size)
    if G == 0:
        return v
    v = np.clip(v, 0.0, np.inf)
    s = float(np.sum(v))
    if (not np.isfinite(s)) or s <= 0.0:
        return (np.ones((G,), dtype=np.float32) / float(G)).astype(np.float32, copy=False)
    return (v / s).astype(np.float32, copy=False)

def sigma_target_from_y_hat(y_hat: np.ndarray, params: object, flags: object) -> np.ndarray:
    """Construct a sigma preference (G,) monotone in y_hat and clipped to [0,sigma_max]."""
    _ = flags
    y = normalize_to_simplex_nonneg(y_hat)
    sigma_max = float(getattr(params, "sigma_max", 1.0))
    if not np.isfinite(sigma_max) or sigma_max <= 0:
        sigma_max = 1.0
    sig = sigma_max * y
    return np.clip(sig, 0.0, sigma_max).astype(np.float32, copy=False)


def sigma_lower_bound_from_y_hat(y_hat: np.ndarray, params: object, flags: object) -> np.ndarray:
    """Compute coupled lower bound lo(G,) for sigma based on (projected) y_hat."""
    _ = flags
    y = normalize_to_simplex_nonneg(y_hat)
    sigma_max = float(getattr(params, "sigma_max", 1.0))
    if not np.isfinite(sigma_max) or sigma_max <= 0:
        sigma_max = 1.0
    ratio = float(getattr(params, "sigma_lb_ratio", 0.0))
    if not np.isfinite(ratio):
        ratio = 0.0
    ratio = float(np.clip(ratio, 0.0, 1.0))
    lo = ratio * sigma_max * y
    return np.clip(lo, 0.0, sigma_max).astype(np.float32, copy=False)
