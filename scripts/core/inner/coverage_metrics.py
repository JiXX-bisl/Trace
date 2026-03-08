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

def normalize_to_box01_nonneg(vec: np.ndarray) -> np.ndarray:
    """
    Clip negatives to 0 then rescale to [0,1] by max (NOT simplex).
    Useful when theory_y_hat_box_only=True.
    """
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    if v.size == 0:
        return v
    v = np.clip(v, 0.0, np.inf)
    m = float(np.max(v)) if v.size else 0.0
    if (not np.isfinite(m)) or m <= 0.0:
        return np.ones_like(v, dtype=np.float32)
    return (v / m).astype(np.float32, copy=False)


def expand_y_hat_time_stacked(
    y_g: np.ndarray,
    *,
    G: int,
    T: int,
    simplex_total: bool,
) -> np.ndarray:
    """
    Expand per-group y_g (G,) into time-stacked y_flat (G*T,) using g-major packing:
        y_flat[g*T + t] = y_gt[g, t]

    If simplex_total=True, we distribute y_g across time so that sum(y_flat)=1.
    If simplex_total=False (box semantics), we repeat across time (each entry in [0,1]).
    """
    y_g = np.asarray(y_g, dtype=np.float32).reshape(-1)
    if G <= 0:
        return np.zeros((0,), dtype=np.float32)
    if y_g.size != G:
        # best-effort truncate/pad
        if y_g.size > G:
            y_g = y_g[:G]
        else:
            y_g = np.pad(y_g, (0, G - y_g.size))
    T = int(T)
    if T <= 1:
        return y_g.astype(np.float32, copy=False)
    if simplex_total:
        y_gt = (y_g[:, None] / float(T)).astype(np.float32)
    else:
        y_gt = np.tile(y_g[:, None], (1, T)).astype(np.float32)
    return y_gt.reshape(-1).astype(np.float32, copy=False)


def _reduce_time_stacked_y_hat(
    y_hat: np.ndarray,
    *,
    G: int,
    reduce: str = "max",
) -> np.ndarray:
    """Reduce y_hat (G*T,) -> (G,) by max/mean/sum over time."""
    y = np.asarray(y_hat, dtype=np.float32).reshape(-1)
    if G <= 0 or y.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if y.size == G:
        return y
    if (y.size % G) != 0:
        # fallback: treat as already group-wise
        return y[:G] if y.size > G else np.pad(y, (0, G - y.size))
    T = int(y.size // G)
    yy = y.reshape((G, T))
    r = str(reduce).lower()
    if r == "mean":
        return np.mean(yy, axis=1).astype(np.float32)
    if r == "sum":
        return np.sum(yy, axis=1).astype(np.float32)
    # default "max"
    return np.max(yy, axis=1).astype(np.float32)


def candidate_y_features(
    problem: object,
    rid: int,
    candidates: np.ndarray,
    reg: object,
    flags: object,
) -> np.ndarray:
    """Build candidate->y_hat feature matrix Phi_y for Route-B theta-QP.

    Returns
    -------
    Phi_y : np.ndarray
        Shape (dy, M_use), where:
          dy = size(y_hat) = G*T (time-stacked) or G (T=1),
          M_use = number of candidates (len(candidates)).

    Convention
    ----------
    y_hat is packed in g-major order for time stacking:
      y_flat[g*T + t] corresponds to y_gt[g, t], where y_gt has shape (G, T).
    """
    cand = np.asarray(candidates, dtype=np.float32)
    if cand.ndim != 2:
        cand = cand.reshape(-1, int(getattr(problem, "coord_dim", 2)))

    M_use = int(cand.shape[0])
    G = int(getattr(problem, "G", 0) or 0)
    if (M_use <= 0) or (G <= 0):
        return np.zeros((0, M_use), dtype=np.float32)

    # dy = size of y_hat block (supports time-stacked G*T)
    try:
        dy = int(np.prod(getattr(reg, "shape")("y_hat")))
    except Exception:
        # fallback: try reg.shape("y_hat")
        try:
            dy = int(np.prod(reg.shape("y_hat")))
        except Exception:
            dy = G

    if dy <= 0:
        return np.zeros((0, M_use), dtype=np.float32)

    if dy % G != 0:
        raise AssertionError(f"y_hat dim dy={dy} must be divisible by G={G} for time stacking.")
    T = int(dy // G)

    theory_mode = bool(getattr(flags, "enable_theory_mode", False))
    y_box_only = bool(getattr(flags, "theory_y_hat_box_only", False))
    use_box = theory_mode and y_box_only

    Phi = np.zeros((dy, M_use), dtype=np.float32)
    for m in range(M_use):
        raw_scores = coverage_group_scores(problem, rid, cand[m])
        y_g = normalize_to_box01_nonneg(raw_scores) if use_box else normalize_to_simplex_nonneg(raw_scores)
        phi_m = expand_y_hat_time_stacked(y_g, G=G, T=T, simplex_total=(not use_box))
        Phi[:, m] = np.asarray(phi_m, dtype=np.float32).reshape(dy)

    return Phi

def theta_to_s(Phi: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Compute local coverage contribution s = Phi @ theta.

    This is a pure helper for readability in Route-B/Y-avg assembly:
      - Phi has shape (dy, M_use)
      - theta has shape (M_use,)
      - returns s with shape (dy,)

    NOTE: This function does not project/normalize; caller is responsible for simplex constraints on theta.
    """
    Phi = np.asarray(Phi, dtype=np.float32)
    th = np.asarray(theta, dtype=np.float32).reshape(-1)
    return (Phi @ th).astype(np.float32, copy=False).reshape(-1)


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
    # _ = flags
    # y = normalize_to_simplex_nonneg(y_hat)
    """Construct a sigma preference (G,) monotone in y_hat and clipped to [0,sigma_max].

    Supports time-stacked y_hat (G*T,) when G is provided via flags/problem.
    """
    G = int(getattr(flags, "G", 0) or 0)  # usually absent; q_step should pass G via flags if needed
    reduce = str(getattr(flags, "sigma_y_reduce", "max"))
    if G > 0:
        y_g = _reduce_time_stacked_y_hat(y_hat, G=G, reduce=reduce)
    else:
        y_g = np.asarray(y_hat, dtype=np.float32).reshape(-1)
    # normalization depends on theory box semantics
    use_box = bool(getattr(flags, "enable_theory_mode", False)) and bool(getattr(flags, "theory_y_hat_box_only", False))
    y = normalize_to_box01_nonneg(y_g) if use_box else normalize_to_simplex_nonneg(y_g)
    sigma_max = float(getattr(params, "sigma_max", 1.0))
    if not np.isfinite(sigma_max) or sigma_max <= 0:
        sigma_max = 1.0
    sig = sigma_max * y
    return np.clip(sig, 0.0, sigma_max).astype(np.float32, copy=False)


def sigma_lower_bound_from_y_hat(y_hat: np.ndarray, params: object, flags: object) -> np.ndarray:
    # """Compute coupled lower bound lo(G,) for sigma based on (projected) y_hat."""
    # _ = flags
    # y = normalize_to_simplex_nonneg(y_hat)
    """Compute coupled lower bound lo(G,) for sigma based on (projected) y_hat.

    Supports time-stacked y_hat (G*T,) when G is provided.
    """
    G = int(getattr(flags, "G", 0) or 0)
    reduce = str(getattr(flags, "sigma_y_reduce", "max"))
    if G > 0:
        y_g = _reduce_time_stacked_y_hat(y_hat, G=G, reduce=reduce)
    else:
        y_g = np.asarray(y_hat, dtype=np.float32).reshape(-1)
    use_box = bool(getattr(flags, "enable_theory_mode", False)) and bool(getattr(flags, "theory_y_hat_box_only", False))
    y = normalize_to_box01_nonneg(y_g) if use_box else normalize_to_simplex_nonneg(y_g)    
    sigma_max = float(getattr(params, "sigma_max", 1.0))
    if not np.isfinite(sigma_max) or sigma_max <= 0:
        sigma_max = 1.0
    ratio = float(getattr(params, "sigma_lb_ratio", 0.0))
    if not np.isfinite(ratio):
        ratio = 0.0
    ratio = float(np.clip(ratio, 0.0, 1.0))
    lo = ratio * sigma_max * y
    return np.clip(lo, 0.0, sigma_max).astype(np.float32, copy=False)
