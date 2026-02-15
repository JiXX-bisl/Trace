"""scripts.core.inner.routing_state

Stage D: Routing state for delta reachability with memory.

We maintain, on ``window_state`` via setattr:
  - parent: (N,) int32, parent node id in current BFS tree (-1 if none)
  - dist_to_root: (N,) int32, hop distance in current BFS tree (-1 if unreachable)
  - last_success_hops: (N,) int32, last known successful hop distance
  - last_success_step: (N,) int32, step index when last_success_hops was refreshed

This aligns with "recently successful forward" semantics rather than recomputing
delta from scratch at each step.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np


def _fresh_edge_mask(link_current: Any, window_state: Any, flags: Any) -> np.ndarray:
    edges = np.asarray(getattr(link_current, "edges", np.zeros((0, 2), dtype=np.int32)), dtype=np.int32).reshape(-1, 2)
    E = int(edges.shape[0])
    if E == 0:
        return np.zeros((0,), dtype=bool)

    if not bool(getattr(flags, "reachability_requires_fresh", True)):
        return np.ones((E,), dtype=bool)

    is_stale = getattr(window_state, "is_stale", None)
    if is_stale is None:
        # if staleness isn't tracked, treat as fresh
        return np.ones((E,), dtype=bool)
    is_stale = np.asarray(is_stale, dtype=bool).reshape(-1)
    if is_stale.size != E:
        return np.ones((E,), dtype=bool)
    return ~is_stale


def _build_adj(N: int, edges: np.ndarray, keep: np.ndarray) -> list[list[int]]:
    adj = [[] for _ in range(N)]
    if edges.size == 0:
        return adj
    keep = np.asarray(keep, dtype=bool).reshape(-1)
    for e, ok in enumerate(keep):
        if not bool(ok):
            continue
        u = int(edges[e, 0])
        v = int(edges[e, 1])
        if 0 <= u < N and 0 <= v < N:
            adj[u].append(v)
            adj[v].append(u)
    return adj


def bfs_tree(N: int, edges: np.ndarray, keep: np.ndarray, root_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (parent, dist) arrays for BFS tree on an undirected graph."""
    parent = -np.ones((N,), dtype=np.int32)
    dist = -np.ones((N,), dtype=np.int32)
    if N <= 0:
        return parent, dist
    root = int(root_id)
    if root < 0 or root >= N:
        root = 0
    adj = _build_adj(N, edges, keep)
    q = [root]
    dist[root] = 0
    parent[root] = root
    head = 0
    while head < len(q):
        u = q[head]
        head += 1
        du = int(dist[u])
        for v in adj[u]:
            if dist[v] >= 0:
                continue
            dist[v] = du + 1
            parent[v] = u
            q.append(v)
    return parent, dist


def build_or_update_tree(step: int, link_current: Any, window_state: Any, root_id: int, flags: Any) -> None:
    """Build BFS tree on current (fresh) subgraph and store to window_state."""
    _ = step
    N = int(getattr(link_current, "robot_ids", None).__len__() if getattr(link_current, "robot_ids", None) is not None else 0)
    # Prefer window_state.frozen_link robot_ids if available
    try:
        N = int(len(getattr(getattr(window_state, "frozen_link", None), "robot_ids", []))) or N
    except Exception:
        pass
    if N <= 0:
        N = int(getattr(window_state, "N", 0) or 0)

    edges = np.asarray(getattr(link_current, "edges", np.zeros((0, 2), dtype=np.int32)), dtype=np.int32).reshape(-1, 2)
    keep = _fresh_edge_mask(link_current, window_state, flags)
    parent, dist = bfs_tree(N, edges, keep, int(root_id))
    setattr(window_state, "parent", parent)
    setattr(window_state, "dist_to_root", dist)


def update_last_successful_forward(step: int, window_state: Any, dist_to_root: np.ndarray, flags: Any) -> None:
    """Refresh last_success_* if node currently has a valid dist_to_root."""
    _ = flags
    dist = np.asarray(dist_to_root, dtype=np.int32).reshape(-1)
    N = int(dist.size)
    if N == 0:
        return
    last_hops = getattr(window_state, "last_success_hops", None)
    last_step = getattr(window_state, "last_success_step", None)
    if last_hops is None or np.asarray(last_hops).shape != (N,):
        last_hops = np.full((N,), 10**9, dtype=np.int32)
    else:
        last_hops = np.asarray(last_hops, dtype=np.int32).copy()

    if last_step is None or np.asarray(last_step).shape != (N,):
        last_step = np.full((N,), -10**9, dtype=np.int32)
    else:
        last_step = np.asarray(last_step, dtype=np.int32).copy()

    for i in range(N):
        d = int(dist[i])
        if d >= 0:
            last_hops[i] = d
            last_step[i] = int(step)

    setattr(window_state, "last_success_hops", last_hops)
    setattr(window_state, "last_success_step", last_step)


def compute_delta(step: int, window_state: Any, max_hops: int, decay: int, flags: Any) -> np.ndarray:
    """Compute δ_i based on last_success_hops/step.

    δ_i=True iff last_success_hops[i] <= max_hops and (if decay>0) step-last_success_step[i] <= decay.
    """
    _ = flags
    max_h = int(max_hops)
    dec = int(decay)
    last_hops = np.asarray(getattr(window_state, "last_success_hops", np.zeros((0,), dtype=np.int32)), dtype=np.int32).reshape(-1)
    last_step = np.asarray(getattr(window_state, "last_success_step", np.zeros((0,), dtype=np.int32)), dtype=np.int32).reshape(-1)
    N = int(last_hops.size)
    if N == 0:
        return np.zeros((0,), dtype=bool)
    if last_step.size != N:
        last_step = np.full((N,), -10**9, dtype=np.int32)
    ok = last_hops <= max_h
    if dec > 0:
        ok = ok & ((int(step) - last_step) <= dec)
    return np.asarray(ok, dtype=bool)
