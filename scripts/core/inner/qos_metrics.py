"""scripts.core.inner.qos_metrics

QoS-aware metrics and local resource allocation helpers.

This module is intentionally **pure** (no solver/ADMM state, no side effects).
It provides small, testable building blocks used by q-step to produce *local*
recommendations for edge allocations (B_hat / f_hat) and a per-robot QoS
violation summary (r_hat).

If you later plug in a propagation model (e.g., Lee) or a learned predictor,
you should only need to replace ``edge_quality_weights`` and/or
``connectivity_score_from_positions`` while keeping call sites stable.
"""

from __future__ import annotations

import numpy as np

def incident_edge_indices(edges: np.ndarray, rid: int) -> np.ndarray:
    """
    Return indices for edges incident to robot ris.

    Parameters
    edges: (E, 2) int array.
    rid: robot id.

    Returns
    np.ndarray (K,) int64 indices.
    """
    try:
        e = np.asarray(edges)
        if e.ndim != 2 or e.shape[1] != 2:
            return np.zeros((0,), dtype=np.int64)
        rid_i = int(rid)
        mask = (e[:, 0] == rid_i) | (e[:, 1] == rid_i)
        return np.nonzero(mask)[0].astype(np.int64, copy=False)
    except Exception:
        return np.zeros((0,), dtype=np.int64)

def _safe_array(x: object, length: int, default: float=0.0) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size == length:
        return arr
    if arr.size == 0:
        return np.full((length,), float(default), dtype=np.float32)
    if arr.size == 1:
        return np.full((length,), float(arr[0]), dtype=np.float32)
    out = np.full((length,), float(default), dtype=np.float32)
    m = int(min(length, arr.size))
    out[:m] = arr[:m]
    return out

def edge_quality_weights(
    link: object,
    params: object,
    flags: object,
    rid: int,
    incident_idx: np.ndarray
) -> np.ndarray:
    """
    Compute nonnegative edge-quality weigths for incident edges
    Higher weight means a better edge: higher capacity, lower delay, lower packet loss rate
    All parameter reads use getattr(..., defualt)
    """
    _ = rid
    idx = np.asarray(incident_idx, dtype=np.int64).reshape(-1)
    if idx.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if not bool(getattr(flags, "enable_qos_aware_edges", True)):
        return np.ones((idx.size,), dtype=np.float32)
    
    need_len = int(np.max(idx)) + 1
    cap_all = _safe_array(getattr(link, "capacity", np.zeros((0,), dtype=np.float32)), need_len)
    dly_all = _safe_array(getattr(link, "delay", np.zeros((0,), dtype=np.float32)), need_len)
    plr_all = _safe_array(getattr(link, "plr", np.zeros((0,), dtype=np.float32)), need_len)

    cap = cap_all[idx]
    delay = dly_all[idx]
    plr = plr_all[idx]

    delay_t = float(getattr(params, "qos_delay_target", 1.0))
    plr_t = float(getattr(params, "qos_plr_target", 0.1))
    wc = float(getattr(params, "qos_weight_capacity", 1.0))
    wd = float(getattr(params, "qos_weight_delay", 1.0))
    wp = float(getattr(params, "qos_weight_plr", 1.0))

    eps = 1e-6
    cap_norm = cap / (float(np.max(cap)) + eps) if cap.size else cap
    delay_term = delay_t / (delay + delay_t + eps)  # (0, 1]
    plr_term = plr_t / (plr + plr_t + eps)          # (0, 1]

    w = wc * cap_norm + wd * delay_term + wp * plr_term
    w = np.clip(w, 0.0, np.inf).astype(np.float32, copy=False)
    if not np.any(w > 0):
        w = np.ones((idx.size,), dtype=np.float32)

    return w

def allocate_over_incident_edges(
    cap_incident: np.ndarray,
    weights: np.ndarray,
    local_budget: float,
) -> np.ndarray:
    """
    Allocate budegt over incident edges respecting per-edge caps.
    Output is nonnegeative and elementwise <= cap_incident.
    """
    cap = np.asarray(cap_incident, dtype=np.float32).reshape(-1)
    w = np.asarray(weights, dtype=np.float32).reshape(-1)
    K = int(min(cap.size, w.size))
    cap = cap[:K]
    w = w[:K]
    if K == 0:
        return np.zeros((0,), dtype=np.float32)
    budget = float(local_budget) if np.isfinite(local_budget) else 0.0
    if budget <= 0.0:
        return np.zeros((K,), dtype=np.float32)
    cap = np.clip(cap, 0.0, np.inf)
    w = np.clip(w, 0.0, np.inf)
    if float(np.sum(cap)) <= budget:
        return cap.astype(np.float32, copy=False)
    
    alloc = np.zeros((K,), dtype=np.float32)
    remaining = float(budget)
    active = (cap > 0) & (w > 0)
    if not np.any(active):
        return alloc
    
    # water filling with proportional shares under caps
    while remaining > 1e-8 and np.any(active):
        w_act = w[active]
        s = float(np.sum(w_act))
        if s <= 0:
            break
        share = (remaining * (w_act / s)).astype(np.float32)
        idx_act = np.nonzero(active)[0]
        hit = False
        for j, ii in enumerate(idx_act):
            give = float(share[j])
            room = float(cap[ii] - alloc[ii])
            if give >= room - 1e-8:
                alloc[ii] = cap[ii]
                remaining -= room
                active[ii] = False
                hit = True
            else:
                alloc[ii] += give
        if not hit:
            remaining = 0.0
    return np.clip(alloc, 0.0, cap).astype(np.float32, copy=False)

def summarize_qos_violation(
    link,
    alloc_full_E: np.ndarray,
    params,
    flags,
    rid: int,
) -> float:
    """
    Return scalar in [0,1]. Larger => worse QoS (more violation).
    Must be monotone w.r.t. stricter targets (smaller delay_target / plr_target -> larger violation).
    """
    # --- safe edge list ---
    edges = np.asarray(getattr(link, "edges", np.zeros((0, 2), dtype=np.int32)))
    if edges.ndim != 2 or edges.shape[1] != 2:
        return 0.0
    E = int(edges.shape[0])
    if E <= 0:
        return 0.0

    idx = incident_edge_indices(edges, rid)
    if idx.size == 0:
        return 0.0

    # --- read targets (CRITICAL: must match test's setattr names) ---
    delay_target = float(getattr(params, "qos_delay_target", 1.0)) if params is not None else 1.0
    plr_target = float(getattr(params, "qos_plr_target", 0.1)) if params is not None else 0.1
    sens = float(getattr(params, "qos_violation_sensitivity", 1.0)) if params is not None else 1.0
    if not np.isfinite(delay_target) or delay_target <= 1e-9:
        delay_target = 1.0
    if not np.isfinite(plr_target) or plr_target <= 1e-9:
        plr_target = 0.1
    if not np.isfinite(sens) or sens <= 1e-9:
        sens = 1.0

    # --- safe arrays ---
    alloc = np.asarray(alloc_full_E, dtype=np.float32).reshape(-1)
    if alloc.size != E:
        # resize without crashing
        if alloc.size == 0:
            alloc = np.zeros((E,), dtype=np.float32)
        elif alloc.size == 1:
            alloc = np.full((E,), float(alloc[0]), dtype=np.float32)
        else:
            alloc = alloc[:E]
            if alloc.size < E:
                alloc = np.pad(alloc, (0, E - alloc.size))

    delay_all = np.asarray(getattr(link, "delay", np.zeros((E,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    if delay_all.size != E:
        if delay_all.size == 0:
            delay_all = np.zeros((E,), dtype=np.float32)
        elif delay_all.size == 1:
            delay_all = np.full((E,), float(delay_all[0]), dtype=np.float32)
        else:
            delay_all = delay_all[:E]
            if delay_all.size < E:
                delay_all = np.pad(delay_all, (0, E - delay_all.size))

    plr_all = np.asarray(getattr(link, "plr", np.zeros((E,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    if plr_all.size != E:
        if plr_all.size == 0:
            plr_all = np.zeros((E,), dtype=np.float32)
        elif plr_all.size == 1:
            plr_all = np.full((E,), float(plr_all[0]), dtype=np.float32)
        else:
            plr_all = plr_all[:E]
            if plr_all.size < E:
                plr_all = np.pad(plr_all, (0, E - plr_all.size))

    # --- violation per incident edge ---
    d = delay_all[idx]
    p = plr_all[idx]

    # ratio-based excess: max(0, delay/delay_target - 1), max(0, plr/plr_target - 1)
    d_excess = np.maximum(d / delay_target - 1.0, 0.0)
    p_excess = np.maximum(p / plr_target - 1.0, 0.0)

    v = d_excess + p_excess
    # map to [0,1]: 1-exp(-sens*v)
    v01 = 1.0 - np.exp(-sens * v)
    v01 = np.clip(v01, 0.0, 1.0)

    # --- aggregate with allocation weights (if alloc is all zero, use mean) ---
    a = np.clip(alloc[idx], 0.0, None)
    s = float(np.sum(a))
    if s > 1e-9:
        out = float(np.sum((a / s) * v01))
    else:
        out = float(np.mean(v01)) if v01.size > 0 else 0.0

    if not np.isfinite(out):
        return 0.0
    return float(np.clip(out, 0.0, 1.0))

def connectivity_score_from_positions(
    candidate_xy: np.ndarray,
    robot_pos: np.ndarray,
    params: object
) -> float:
    """
    Fallback connectivity tendency based only on geometry

    Returns a *cost-like* scalar (lower is better) penalizing being far from neighbors.
    Intended to be added into candidate score when enable_qstep_cost_qos_pos is True.
    """
    try: 
        cand = np.asarray(candidate_xy, dtype=np.float32).reshape(2)
        rp = np.asarray(robot_pos, dtype=np.float32).reshape(-1, 2)
    except Exception:
        return 0.0

    N = int(rp.shape[0])
    if N <= 1:
        return 0.0
    
    radius = float(getattr(params, "qos_pos_neighbor_radius", 10.0))
    weight = float(getattr(params, "qos_pos_neighbot_weight", 1.0))
    if not np.isfinite(radius) or radius <= 0:
        radius = 10.0
    if not np.isfinite(weight):
        weight = 1.0

    d = np.linalg.norm(rp - cand[None, :], axis=1)
    d_sorted = np.sort(d)
    d_other = d_sorted[1:] if d_sorted.size > 1 else np.asarray([], dtype=np.float32)
    if d_other.size == 0:
        return 0.0
    min_other = float(d_other[0])

    penalty = max(0.0, (min_other - radius) / radius)
    n_in = int(np.sum(d_other <= radius))
    cost = weight * penalty - weight * 0.1 * float(n_in)
    if not np.isfinite(cost):
        return 0.0
    return float(cost)

def normalize_budget(
    usage: np.ndarray,
    window_state,
    *,
    mode: str = "per_edge_ref",
    eps: float = 1e-8
) -> np.ndarray:
    """
    Normalize budget/usage to a comparable scale.

    - per_edge_ref: usage / ref_rate (edge-wise)
    - global_ref: sum(usage) / ref_total (scalar broadcast)
    """
    u = np.asarray(usage, dtype=np.float32).reshape(-1)
    ref = getattr(window_state, "ref_rate", None)
    if ref is None:
        try:
            ref = np.asarray(getattr(getattr(window_state, "frozen_link", None), "capacity", None), dtype=np.float32)
        except Exception:
            ref = None
    if ref is None:
        ref = np.ones_like(u)
    ref = np.asarray(ref, dtype=np.float32).reshape(-1)
    if ref.size != u.size:
        ref = np.ones_like(u)
    if mode == "global_ref":
        ref_total = getattr(window_state, "ref_total", None)
        if ref_total is None:
            ref_total = float(np.sum(ref))
        denom = float(ref_total) if float(ref_total) > float(eps) else 1.0
        ratio = float(np.sum(u)) / denom
        return np.full_like(u, ratio, dtype=np.float32)
    denom = np.maximum(ref, float(eps))
    return (u / denom).astype(np.float32)

def update_budget_cache(
    *,
    window_state,
    reg,
    value: np.ndarray,
    ema: float,
    enable_update: bool,
    enable_soft_violation: bool,
    tol: float,
    flags = None,
) -> None:
    """Update window_state.budget_cache and optional soft violation.

    This is intentionally soft: it never changes feasibility/projections.

    Parameters
    ----------
    value:
        If 1D: treated as z.
        If 2D: treated as q (N,D), aggregated by mean.
    """
    _ = flags
    if window_state is None:
        return
    try:
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 1:
            z_blocks = reg.unpack(arr)
        elif arr.ndim == 2:
            z_blocks = reg.unpack(np.mean(arr, axis=0))
        else:
            return

        # Budget cache should track *byte budget usage* by default.
        # Prefer B_hat (bytes), fallback to f_hat (flow) if B_hat is unavailable.
        if "B_hat" in z_blocks:
            usage = np.asarray(z_blocks["B_hat"], dtype=np.float32).reshape(-1)
        elif "f_hat" in z_blocks:
            usage = np.asarray(z_blocks["f_hat"], dtype=np.float32).reshape(-1)
        else:
            return
    except Exception:
        return

    def _edge_key(i: int, j: int) -> tuple[int, int]:
        return (i, j) if i <= j else (j, i)

    edges = np.asarray(getattr(getattr(window_state, "frozen_link", None), "edges", np.zeros((0, 2), np.int32)), dtype=np.int32).reshape(-1, 2)
    E = int(edges.shape[0])
    if usage.size != E:
        # Nothing to update.
        return

    cache_obj = getattr(window_state, "budget_cache", None)
    if not isinstance(cache_obj, dict):
        cache_obj = {}

    a = float(ema)
    a = min(max(a, 0.0), 1.0)
    cache_vals = np.zeros((E,), dtype=np.float32)

    for k, (i, j) in enumerate(edges):
        key = _edge_key(int(i), int(j))
        prev = float(cache_obj.get(key, 0.0))
        cur = float(usage[k])
        newv = cur if (not enable_update or a == 0.0) else (a * prev + (1.0 - a) * cur)
        if enable_update:
            cache_obj[key] = float(newv)
        cache_vals[k] = float(cache_obj.get(key, prev if not enable_update else newv))

    if enable_update:
        setattr(window_state, "budget_cache", cache_obj)

    if enable_soft_violation:
        ref = getattr(window_state, "ref_rate", None)
        if ref is None:
            ref = np.asarray(getattr(getattr(window_state, "frozen_link", None), "capacity", np.ones((E,), np.float32)), dtype=np.float32).reshape(-1)
        ref = np.asarray(ref, dtype=np.float32).reshape(-1)
        if ref.size != E:
            ref = np.ones((E,), dtype=np.float32)
        viol = np.maximum(cache_vals - (ref - float(tol)), 0.0)
        setattr(window_state, "budget_violation_vec", np.asarray(viol, dtype=np.float32))
        setattr(window_state, "budget_violation", float(np.max(viol)) if viol.size > 0 else 0.0)

