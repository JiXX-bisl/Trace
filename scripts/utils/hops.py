import numpy as np
from collections import deque
from typing import Callable, Dict, Any, Tuple, List
import heapq

INF_I32 = np.int32(10**9)
INF_F = 1e30
INF_H = 10**9
INF_L = 1e30

def compute_comm_undirected_minlen(
    pos: np.ndarray,
    los_max_dist: float,
    is_los_fn: Callable[[np.ndarray, np.ndarray], bool],
    C_max: float,
) -> Dict[str, Any]:
    """
    无向 LoS 图 + 全源最小跳数 + 在最小跳数约束下选 path_len 最短的路径计算 cap.

    Parameters
    ----------
    pos : (N,3) float
    los_max_dist : float
    is_los_fn : (p_i, p_j) -> bool   # 必须对称(或你认为LoS物理上对称)
    C_max : float

    Returns
    -------
    dict with:
      adj: (N,N) bool
      hops: (N,N) int32   # 不可达为 -1
      path_len: (N,N) float32   # 不可达为 0
      cap: (N,N) float32        # 不可达为 0
      pos: (N,3) float32
    """
    pos = np.asarray(pos, dtype=np.float32)
    N = pos.shape[0]

    # ---------
    # 1) Build undirected adjacency
    # ---------
    adj = np.zeros((N, N), dtype=np.bool_)
    dist_mat = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)

    for i in range(N):
        for j in range(i + 1, N):
            if dist_mat[i, j] > float(los_max_dist):
                continue
            if is_los_fn(pos[i], pos[j]):
                adj[i, j] = True
                adj[j, i] = True

    # adjacency list (faster & cleaner than np.where in loops)
    nbrs = [np.flatnonzero(adj[i]).astype(np.int32) for i in range(N)]

    # ---------
    # 2) For each source s:
    #    (a) BFS for minimal hops
    #    (b) DP on hop-layers to get minimal path_len among minimal-hop paths
    # ---------
    hops = np.full((N, N), INF_I32, dtype=np.int32)
    path_len = np.full((N, N), INF_F, dtype=np.float32)
    prev = np.full((N, N), -1, dtype=np.int32)  # prev[s, v] to reconstruct chosen path if needed

    for s in range(N):
        # (a) BFS hops
        hops[s, s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            du = int(hops[s, u])
            for v in nbrs[u]:
                if hops[s, v] == INF_I32:  # first time reached => minimal hop
                    hops[s, v] = du + 1
                    q.append(int(v))

        # (b) DP for minimal path_len given minimal hops
        # Any minimal-hop path increases hop by exactly +1 each step.
        path_len[s, s] = 0.0
        max_d = int(hops[s].min(initial=0))  # not used; we'll compute actual max over reachable
        reachable = np.where(hops[s] < INF_I32)[0]
        if reachable.size == 0:
            continue
        max_d = int(hops[s, reachable].max())

        # process layer by layer
        for d in range(max_d):
            layer_u = np.where(hops[s] == d)[0]
            if layer_u.size == 0:
                continue
            for u in layer_u:
                if float(path_len[s, u]) >= INF_F / 2:
                    continue
                pu_len = float(path_len[s, u])
                for v in nbrs[u]:
                    v = int(v)
                    # only allow forward edges that preserve minimal hops
                    if int(hops[s, v]) != d + 1:
                        continue
                    cand = pu_len + float(np.linalg.norm(pos[u] - pos[v]))
                    if cand < float(path_len[s, v]) - 1e-9:
                        path_len[s, v] = cand
                        prev[s, v] = int(u)

    # ---------
    # 3) cap from hops + path_len
    # ---------
    cap = np.zeros((N, N), dtype=np.float32)
    for s in range(N):
        for t in range(N):
            if t == s:
                continue
            nh = int(hops[s, t])
            if nh == INF_I32 or nh <= 0:
                continue
            plen = float(path_len[s, t])
            if not np.isfinite(plen) or plen <= 1e-6 or plen >= INF_F / 2:
                continue
            cap[s, t] = float(C_max) / float(plen * nh)

    # export hops with -1 for unreachable, and path_len with 0 for unreachable (optional)
    hops_out = hops.copy()
    hops_out[hops_out >= INF_I32] = -1

    path_len_out = path_len.copy()
    path_len_out[~np.isfinite(path_len_out)] = 0.0
    path_len_out[path_len_out >= INF_F / 2] = 0.0

    return {
        "adj": adj,
        "hops": hops_out.astype(np.int32),
        "path_len": path_len_out.astype(np.float32),
        "cap": cap.astype(np.float32),
        "pos": pos,
        # 如果你之后想回溯路径可用：
        "prev": prev,
    }


def capacity_for_rid_candidates_minlen(
    rid: int,
    candidates: np.ndarray,
    base_pos: np.ndarray,
    los_max_dist: float,
    is_los_fn: Callable[[np.ndarray, np.ndarray], bool],
    C_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对每个 candidate 位置，把 base_pos[rid] 替换后重算全图通信，
    返回 rid -> all 的 cap/hops.

    Parameters
    ----------
    rid : int
    candidates : (M,3) or (M,2)
    base_pos : (N,3)
    Returns
    -------
    caps_to : (M,N) float32   # caps_to[k, j] = cap(rid->j) under candidate k
    hops_to : (M,N) int32     # -1 unreachable
    """
    base_pos = np.asarray(base_pos, dtype=np.float32)
    cand = np.asarray(candidates, dtype=np.float32)
    N = base_pos.shape[0]
    M = cand.shape[0]

    if cand.ndim != 2 or cand.shape[0] == 0:
        return np.zeros((0, N), np.float32), np.zeros((0, N), np.int32)

    # If candidates are (x,y), pad z using current z of rid
    if cand.shape[1] == 2 and base_pos.shape[1] == 3:
        z0 = float(base_pos[rid, 2])
        cand = np.concatenate([cand, np.full((M, 1), z0, dtype=np.float32)], axis=1)

    if cand.shape[1] != base_pos.shape[1]:
        raise ValueError(f"candidates dim {cand.shape[1]} != base_pos dim {base_pos.shape[1]}")

    caps_to = np.zeros((M, N), dtype=np.float32)
    hops_to = np.full((M, N), -1, dtype=np.int32)

    for k in range(M):
        pos_new = base_pos.copy()
        pos_new[rid] = cand[k]
        out = compute_comm_undirected_minlen(pos_new, los_max_dist, is_los_fn, C_max)
        caps_to[k] = out["cap"][rid]
        hops_to[k] = out["hops"][rid]

    return caps_to, hops_to


# ======= 速度优化 =======

def _build_fixed_graph_excluding_rid(
    *,
    rid: int,
    base_pos: np.ndarray,
    los_max_dist: float,
    is_los_fn: Callable[[np.ndarray, np.ndarray], bool],
) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
    """
    Precompute adjacency/edge-length among nodes excluding rid.
    Returns:
      adj_fixed: (N,N) bool
      len_fixed: (N,N) float32 (inf where no edge)
      nbr_fixed: neighbors list for each node (excluding rid edges only)
    """
    pos = np.asarray(base_pos, dtype=np.float32)
    N = pos.shape[0]
    adj = np.zeros((N, N), dtype=bool)
    leng = np.full((N, N), np.inf, dtype=np.float32)

    for i in range(N):
        if i == rid:
            continue
        for j in range(i + 1, N):
            if j == rid:
                continue
            d = float(np.linalg.norm(pos[i] - pos[j]))
            if d > float(los_max_dist):
                continue
            if not is_los_fn(pos[i], pos[j]):
                continue
            adj[i, j] = adj[j, i] = True
            leng[i, j] = leng[j, i] = np.float32(d)

    nbr = [[] for _ in range(N)]
    for i in range(N):
        if i == rid:
            continue
        nbr[i] = [j for j in range(N) if j != rid and adj[i, j]]

    # rid has no fixed neighbors here
    nbr[rid] = []
    return adj, leng, nbr

def _single_source_minhop_minlen(
    *,
    src: int,
    pos_new: np.ndarray,
    adj_fixed: np.ndarray,
    len_fixed: np.ndarray,
    nbr_fixed: List[List[int]],
    los_max_dist: float,
    is_los_fn: Callable[[np.ndarray, np.ndarray], bool],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build neighbors for src based on pos_new[src] and fixed graph for others,
    then BFS hops and DP minlen among minhop paths.

    Returns:
      hops: (N,) int32  (-1 unreachable, 0 at src)
      plen: (N,) float32 (inf unreachable, 0 at src)
      cap : (N,) float32 (0 for unreachable/src)
    """
    pos = np.asarray(pos_new, dtype=np.float32)
    N = pos.shape[0]

    # Build src neighbors for this candidate (only N-1 checks)
    src_nbr = []
    src_len = {}
    psrc = pos[src]
    for j in range(N):
        if j == src:
            continue
        d = float(np.linalg.norm(psrc - pos[j]))
        if d > float(los_max_dist):
            continue
        if not is_los_fn(psrc, pos[j]):
            continue
        src_nbr.append(j)
        src_len[j] = np.float32(d)

    # BFS for min-hop from src
    hops = np.full((N,), -1, dtype=np.int32)
    hops[src] = 0
    q = deque([src])
    while q:
        u = q.popleft()
        hu = int(hops[u])

        if u == src:
            neighbors = src_nbr
        else:
            neighbors = nbr_fixed[u] + ([src] if (adj_fixed[u, src]) else [])  # usually false; safe
            # (We won't rely on u->src fixed; src edges handled by src_nbr only)
            # Keep as fixed-only to avoid double-handling.

        for v in neighbors:
            if v == src:
                continue
            if hops[v] == -1:
                hops[v] = hu + 1
                q.append(v)

    # DP: min path length among paths constrained to hop levels
    plen = np.full((N,), np.inf, dtype=np.float32)
    plen[src] = 0.0

    max_h = int(hops[hops >= 0].max()) if np.any(hops >= 0) else 0
    # Process nodes by hop level
    for h in range(max_h):
        # nodes at level h
        us = np.where(hops == h)[0]
        for u in us:
            du = float(plen[u])
            if not np.isfinite(du):
                continue

            if u == src:
                neighbors = src_nbr
                for v in neighbors:
                    if hops[v] == h + 1:
                        dv = du + float(src_len[v])
                        if dv < float(plen[v]):
                            plen[v] = np.float32(dv)
            else:
                for v in nbr_fixed[u]:
                    if hops[v] == h + 1:
                        w = float(len_fixed[u, v])
                        dv = du + w
                        if dv < float(plen[v]):
                            plen[v] = np.float32(dv)

    # cap computed outside (need C_max)
    cap = np.zeros((N,), dtype=np.float32)
    return hops, plen, cap


def capacity_for_rid_candidates_minlen_src(
    rid: int,
    candidates: np.ndarray,
    base_pos: np.ndarray,
    los_max_dist: float,
    is_los_fn: Callable[[np.ndarray, np.ndarray], bool],
    C_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Faster version:
      - Precompute fixed graph excluding rid once.
      - For each candidate, only compute rid edges + single-source BFS + DP.
    Returns:
      caps_to: (M,N)
      hops_to: (M,N)
    """
    base_pos = np.asarray(base_pos, dtype=np.float32)
    cand = np.asarray(candidates, dtype=np.float32)
    N = int(base_pos.shape[0])

    if cand.ndim != 2 or cand.shape[0] == 0:
        return np.zeros((0, N), np.float32), np.zeros((0, N), np.int32)

    M = int(cand.shape[0])

    # pad z if needed
    if cand.shape[1] == 2 and base_pos.shape[1] == 3:
        z0 = float(base_pos[rid, 2])
        cand = np.concatenate([cand, np.full((M, 1), z0, dtype=np.float32)], axis=1)

    if cand.shape[1] != base_pos.shape[1]:
        raise ValueError(f"candidates dim {cand.shape[1]} != base_pos dim {base_pos.shape[1]}")

    # Precompute fixed graph once per call
    adj_fixed, len_fixed, nbr_fixed = _build_fixed_graph_excluding_rid(
        rid=rid,
        base_pos=base_pos,
        los_max_dist=los_max_dist,
        is_los_fn=is_los_fn,
    )

    caps_to = np.zeros((M, N), dtype=np.float32)
    hops_to = np.full((M, N), -1, dtype=np.int32)

    C_max = float(C_max)
    C_max = max(C_max, 1e-6)

    for k in range(M):
        pos_new = base_pos.copy()
        pos_new[rid] = cand[k]

        hops, plen, _ = _single_source_minhop_minlen(
            src=rid,
            pos_new=pos_new,
            adj_fixed=adj_fixed,
            len_fixed=len_fixed,
            nbr_fixed=nbr_fixed,
            los_max_dist=los_max_dist,
            is_los_fn=is_los_fn,
        )

        # compute cap row from hops & plen
        cap_row = np.zeros((N,), dtype=np.float32)
        for j in range(N):
            hj = int(hops[j])
            if j == rid or hj <= 0:
                cap_row[j] = 0.0
                continue
            pj = float(plen[j])
            if not np.isfinite(pj) or pj <= 1e-9:
                cap_row[j] = 0.0
                continue
            cap_row[j] = np.float32(C_max / (pj * hj))

        caps_to[k] = cap_row
        hops_to[k] = hops

    return caps_to, hops_to

