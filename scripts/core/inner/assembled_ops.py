"""scripts.core.inner.assembled_ops

Stage D: Owner-mask + Assembled averaging

This module provides:
- owner_mask_for_block: (N, dim_block) bool mask indicating which robot(s) "own"
  each coordinate of a block for assembled operations.
- assembled_average: coordinate-wise masked average over robots, optionally using
  active_mask δ.

All features are toggle-able via getattr(flags, ...). Defaults must preserve Stage C.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _safe_N(problem: Any, reg: Any) -> int:
    try:
        return int(getattr(problem, "N", 0))
    except Exception:
        return int(getattr(reg, "N", 0) or 0)


def _edges(problem: Any) -> np.ndarray:
    link = getattr(problem, "link", None)
    edges = getattr(link, "edges", np.zeros((0, 2), dtype=np.int32))
    return np.asarray(edges, dtype=np.int32).reshape(-1, 2)

def _orient_edge(u: int, v: int, root: int, mode: str) -> tuple[int, int]:
    """Deterministically orient an undirected edge {u,v} -> (src,dst).

    This is an *ownership/orientation convention* only. It does NOT change the
    stored edge list length E, so it is safe for backward compatibility.

    mode:
      - "as_is"  : keep (u,v)
      - "min_id" : orient from min(u,v) to max(u,v)  (deterministic)
      - "root"   : if one endpoint is root, orient root -> other, else fallback to min_id
    """
    m = str(mode or "as_is")
    if m == "as_is":
        return int(u), int(v)
    if m == "root":
        if u == root and v != root:
            return int(u), int(v)
        if v == root and u != root:
            return int(v), int(u)
        # fallback
        a = int(min(u, v)); b = int(max(u, v))
        return a, b
    # default: min_id
    a = int(min(u, v)); b = int(max(u, v))
    return a, b


def _root_id(problem: Any, flags: Any) -> int:
    # Prefer reachability root if provided; else allow explicit assembled_root_id;
    # else fallback to params.root_id_default; else 0.
    try:
        params = getattr(problem, "params", None)
    except Exception:
        params = None
    return int(
        getattr(
            flags,
            "reachability_root_id",
            getattr(flags, "assembled_root_id", getattr(params, "root_id_default", 0)),
        )
    )


def owner_mask_for_block(block_name: str, problem: Any, reg: Any, flags: Any) -> np.ndarray:
    """Return owner mask for a block.

    Shape: (N, dim_block). `dim_block` is the flat coordinate dimension for the block.
    Modes (getattr(flags, 'assembled_owner_mode', 'all')):
      - all
      - pos_only
      - edge_by_src
      - edge_by_incident
      - root_only
    """
    N = _safe_N(problem, reg)
    sl = reg.sl(block_name)
    dim = int(sl.stop - sl.start)
    m = np.ones((N, dim), dtype=bool)

    mode = str(getattr(flags, "assembled_owner_mode", "all"))
    if mode == "all":
        return m
    
    if mode == "pos_only":
        if block_name != "pos":
            return m
        # pos is (N,2) flattened -> each robot owns its own 2 coords
        # we added coord flag
        coord_dim = problem.coord_dim
        m[:] = False
        for i in range(N):
            a = coord_dim * i
            b = a + coord_dim
            if b <= dim:
                m[i, a:b] = True
        return m
    
    # ------------------------------------------------------------------
    # M_red support (Table 5.1):
    #   C row-block uses "sum-to" elimination basis to avoid dual drift.
    # Engineering proxy: drop one redundant "all-sum" direction by excluding
    # the root node from owning this block (N-1 basis).
    #
    # Enabled only when requested; default behavior remains unchanged.
    # ------------------------------------------------------------------
    elim_sum_to = bool(getattr(flags, "assembled_eliminate_sum_to_basis", False))
    if elim_sum_to and block_name in ("B_hat",):
        # If B_hat is present, make an (N-1) ownership basis by removing root.
        # This matches "全和等式消元取基" rationale in Table 5.1.
        root = _root_id(problem, flags)
        if 0 <= root < N and N >= 2:
            m[:] = True
            m[root, :] = False
            return m
        # degenerate: keep default
        return m

    if mode in ("edge_by_src", "edge_by_incident"):
        # Only meaningful for edge-shaped blocks. Use registry shape to decide.
        shp = tuple(reg.shape(block_name))
        if len(shp) == 0:
            return m
        # Expect first dim is E for edge blocks.
        E = int(shp[0])
        if E <= 0:
            return np.zeros((N, dim), dtype=bool)

        edges = _edges(problem)
        if edges.shape[0] != E:
            # fallback: treat as all to keep stable
            return m

        m[:] = False
        per_edge_dim = int(dim // E) if E > 0 else 0

        # Table 5.1 requires undirected-edge orientation / de-dup convention.
        # Here we only enforce the *orientation convention* used for ownership.
        # Real de-dup of E should be done upstream (window/staleness) when you
        # construct E_t^->. This keeps backward compatibility.
        orient_undirected = bool(getattr(flags, "assembled_orient_undirected_edges", False)) or bool(
            getattr(flags, "enable_theory_mode", False)
        )
        orient_mode = str(getattr(flags, "assembled_edge_orientation_mode", "as_is"))
        root = _root_id(problem, flags) if orient_undirected else 0

        for e in range(E):
            u = int(edges[e, 0])
            v = int(edges[e, 1])
            if orient_undirected:
                u, v = _orient_edge(u, v, root=root, mode=orient_mode)            
            a = e * per_edge_dim
            b = a + per_edge_dim
            if b > dim:
                break
            if mode == "edge_by_src":
                if 0 <= u < N:
                    m[u, a:b] = True
            else:
                if 0 <= u < N:
                    m[u, a:b] = True
                if 0 <= v < N:
                    m[v, a:b] = True
        return m

    if mode == "root_only":
        m[:] = False
        root = _root_id(problem, flags)
        if 0 <= root < N:
            m[root, :] = True
        return m

    # unknown -> default all
    return np.ones((N, dim), dtype=bool)


def assembled_average(
    block_name: str,
    values_by_robot: np.ndarray,
    owner_mask_row: np.ndarray,
    active_mask: np.ndarray,
    eps: float,
) -> np.ndarray:
    """Coordinate-wise masked average.

    values_by_robot: (N, dim_block)
    owner_mask_row:  (N, dim_block) bool
    active_mask:     (N,) bool

    Average each coordinate over robots where owner==True and (optionally) active==True.
    If a coordinate has no participants, fallback to plain mean over all robots.
    """
    _ = block_name
    v = np.asarray(values_by_robot, dtype=np.float32)
    own = np.asarray(owner_mask_row, dtype=bool)
    act = np.asarray(active_mask, dtype=bool).reshape(-1)
    N, D = v.shape

    if own.shape != (N, D):
        own = np.ones((N, D), dtype=bool)

    if act.shape != (N,):
        act = np.ones((N,), dtype=bool)

    # participants mask: (N, D)
    part = own & act[:, None]

    # weighted sum and count per coordinate
    w = part.astype(np.float32)
    num = np.sum(w * v, axis=0)  # (D,)
    den = np.sum(w, axis=0)      # (D,)

    # fallback to plain mean where den==0
    mean_all = np.mean(v, axis=0)
    out = np.where(den > float(eps), num / (den + float(eps)), mean_all)
    return np.asarray(out, dtype=np.float32)
