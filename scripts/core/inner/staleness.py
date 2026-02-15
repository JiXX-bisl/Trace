from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from scripts.core.data import LinkSnapshot, CommWindowState


def _edge_key(i: int, j: int) -> tuple[int, int]:
    # 推荐无向稳定 key（与你当前 link.edges 可能 i<j 一致，但更稳）
    return (i, j) if i <= j else (j, i)

def update_last_seen(step: int, link_current: LinkSnapshot, window_state: CommWindowState) -> None:
    last_seen = getattr(window_state, "last_seen_step", None)
    if not isinstance(last_seen, dict):
        last_seen = {}
        setattr(window_state, "last_seen_step", last_seen)

    edges = np.asarray(link_current.edges, dtype=np.int32).reshape(-1, 2)
    for (i, j) in edges:
        k = _edge_key(int(i), int(j))
        last_seen[k] = int(step)


def compute_is_stale(
    step: int,
    link_frozen: LinkSnapshot,
    window_state: CommWindowState,
    ttl_steps: int,
    strict: bool = False
) -> np.ndarray:
    """
    Compute stale mask for frozen edges based on last_seen_step and ttl_steps.
    """
    edges = np.asarray(link_frozen.edges, dtype=np.int32).reshape(-1, 2)
    E = edges.shape[0]

    last_seen: Dict[Tuple[int, int], int] = getattr(window_state, "last_seen_step", {})
    base = int(getattr(window_state, "start_step", step))

    out = np.zeros((E,), dtype=bool)
    ttl = int(max(ttl_steps, 0))
    for e, (i, j) in enumerate(edges):
        k = _edge_key(int(i), int(j))
        ls = int(last_seen.get(k, base))
        dt = int(step) - ls
        out[e] = (dt >= ttl) if strict else (dt > ttl)
    return out

def compute_active_hint(N: int, link_used: LinkSnapshot, is_stale: np.ndarray, *, step: int | None = None, 
                        window_state: CommWindowState | None = None, flags = None, params = None) -> np.ndarray:
    """Compute δ/active-hint.

    Default (flags is None or flags.active_hint_mode == "incident"):
      Robots with at least one incident *fresh* edge are marked active.

    Stage D modes:
      - reachability: BFS from root on fresh subgraph; active if dist<=max_hops
      - reachability_with_memory: maintain last_success_hops/step and compute δ
     """
    N = int(N)
    edges = np.asarray(link_used.edges, dtype=np.int32).reshape(-1, 2)
    stale = np.asarray(is_stale, dtype=bool).reshape(-1)
    if edges.shape[0] == 0:
        return np.zeros((N,), dtype=bool)
    stale = stale if stale.shape[0] == edges.shape[0] else np.zeros((edges.shape[0],), dtype=bool)
    fresh = ~stale

    def _incident():
        mask = np.zeros((N,), dtype=bool)
        for (i, j) , s in zip(edges, stale):
            if bool(s): continue
            ii = int(i)
            jj = int(j)
            if 0 <= ii < N:
                mask[ii] = True
            if 0 <= jj < N:
                mask[jj] = True
        return mask
    
    if flags is None:
        return _incident()
    
    mode = str(getattr(flags, "active_hint_mode", "incident"))
    if mode == "incident":
        return _incident()
    
    requires_fresh = bool(getattr(flags, "reachability_requires_fresh", True))
    keep = fresh if requires_fresh else np.ones_like(fresh, dtype=bool)
    root = _choose_root_id(N, edges, keep, flags)

    max_hops = int(getattr(params, "reachability_max_hops", getattr(params, "ttl_hops", 0))) if params is not None else 0
    decay = int(getattr(params, "reachability_memory_decay", 0)) if params is not None else 0

    if mode == "reachability":
        from scripts.core.inner.routing_state import bfs_tree

        _, dist = bfs_tree(N, edges, keep, root)
        hint = (dist >= 0) & (dist <= int(max_hops))
        if bool(getattr(flags, "reachability_fallback_to_incident", True)) and not bool(np.any(hint)):
            return _incident()
        return np.asarray(hint, dtype=bool)

    if mode == "reachability_with_memory":
        if step is None or window_state is None:
            return _incident() if bool(getattr(flags, "reachability_fallback_to_incident", True)) else _incident()
        from scripts.core.inner import routing_state

        # routing_state uses window_state.is_stale
        setattr(window_state, "is_stale", np.asarray(stale, dtype=bool).copy())
        routing_state.build_or_update_tree(int(step), link_used, window_state, root, flags)
        dist = np.asarray(getattr(window_state, "dist_to_root", np.full((N,), -1, dtype=np.int32)), dtype=np.int32)
        routing_state.update_last_successful_forward(int(step), window_state, dist, flags)
        delta = routing_state.compute_delta(int(step), window_state, int(max_hops), int(decay), flags)
        if bool(getattr(flags, "reachability_fallback_to_incident", True)) and not bool(np.any(delta)):
            return _incident()
        return np.asarray(delta, dtype=bool)

    return _incident()


def attach_staleness_to_window(window_state: CommWindowState, is_stale: np.ndarray, active_hint: np.ndarray) -> None:
    setattr(window_state, "is_stale", np.asarray(is_stale, dtype=bool).copy())
    setattr(window_state, "active_hint", np.asarray(active_hint, dtype=bool).copy())

def _choose_root_id(N: int, edges: np.ndarray, keep_mask: np.ndarray, flags) -> int:
    mode = str(getattr(flags, "reachability_root_mode", "base0"))
    if mode == "given":
        return int(getattr(flags, "reachability_root_id", 0))
    if mode == "base0":
        return 0
    if mode == "highest_degree":
        deg = np.zeros((N,), dtype=np.int32)
        keep = np.asarray(keep_mask, dtype=bool).reshape(-1)
        for e, ok in enumerate(keep):
            if not bool(ok):
                continue
            u = int(edges[e, 0])
            v = int(edges[e, 1])
            if 0 <= u < N:
                deg[u] += 1
            if 0 <= v < N:
                deg[v] += 1
        return int(np.argmax(deg)) if N > 0 else 0
    return 0
