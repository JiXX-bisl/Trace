# scripts/core/inner/window.py
from __future__ import annotations

from typing import Optional, Tuple, List
import numpy as np

from scripts.core.data import LinkSnapshot, CommWindowState, FeatureFlags
from scripts.core.inner.staleness import (
    update_last_seen,
    compute_is_stale,
    compute_active_hint,
    attach_staleness_to_window
)

def _should_orient_dedup(flags: FeatureFlags) -> bool:
    # Use the same switch you already introduced for M_red / ownership alignment.
    return bool(getattr(flags, "assembled_orient_undirected_edges", False))


def _root_id_from_flags(flags: FeatureFlags) -> int:
    # Keep consistent with assembled_ops.py's root selection order (best-effort).
    try:
        return int(getattr(flags, "assembled_root_id", 0))
    except Exception:
        return 0


def _orient_edge(u: int, v: int, root: int, mode: str) -> Tuple[int, int]:
    m = str(mode or "min_id")
    if m == "as_is":
        return int(u), int(v)
    if m == "root":
        if u == root and v != root:
            return int(u), int(v)
        if v == root and u != root:
            return int(v), int(u)
        a = int(min(u, v)); b = int(max(u, v))
        return a, b
    # default: min_id
    a = int(min(u, v)); b = int(max(u, v))
    return a, b


def _orient_and_dedup_link(link: LinkSnapshot, flags: FeatureFlags) -> LinkSnapshot:
    """Orient undirected edges deterministically and remove duplicates.

    Output edges are directed keys (src,dst) and arrays are kept aligned.
    Only active when _should_orient_dedup(flags) is True.
    """
    if not _should_orient_dedup(flags):
        return link

    edges = np.asarray(link.edges, dtype=np.int32).reshape(-1, 2)
    E = int(edges.shape[0])
    if E <= 0:
        return link

    root = _root_id_from_flags(flags)
    mode = str(getattr(flags, "assembled_edge_orientation_mode", "min_id"))

    keep_idx: List[int] = []
    new_edges: List[Tuple[int, int]] = []
    seen: set[Tuple[int, int]] = set()
    for e in range(E):
        u = int(edges[e, 0]); v = int(edges[e, 1])
        src, dst = _orient_edge(u, v, root=root, mode=mode)
        key = (src, dst)
        if key in seen:
            continue
        seen.add(key)
        keep_idx.append(e)
        new_edges.append(key)

    keep = np.asarray(keep_idx, dtype=np.int64)
    edges2 = np.asarray(new_edges, dtype=np.int32).reshape(-1, 2)

    def _pick(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim == 0:
            return x
        x = x.reshape(-1)
        if x.size == E:
            return x[keep].copy()
        # fallback: mismatch size, pass through
        return x.copy()

    return LinkSnapshot(
        robot_ids=list(link.robot_ids),
        edges=edges2.copy(),
        signal=_pick(link.signal),
        capacity=_pick(link.capacity),
        delay=_pick(link.delay),
        plr=_pick(link.plr),
        is_stale=_pick(link.is_stale).astype(bool, copy=False),
    )

 



def _clone_link(link: LinkSnapshot) -> LinkSnapshot:
    """Deep-ish copy: arrays are copied; robot_ids list is copied."""
    return LinkSnapshot(
        robot_ids=list(link.robot_ids),
        edges=np.asarray(link.edges).copy(),
        signal=np.asarray(link.signal).copy(),
        capacity=np.asarray(link.capacity).copy(),
        delay=np.asarray(link.delay).copy(),
        plr=np.asarray(link.plr).copy(),
        is_stale=np.asarray(link.is_stale).copy(),
    )


def open_or_update_window(
    step: int,
    link_current: LinkSnapshot,
    window_state: Optional[CommWindowState],
    W: int,
    flags: FeatureFlags,
    rng: np.random.Generator,
) -> CommWindowState:
    """
    Open a new window iff:
      - window_state is None, or
      - step crosses window boundary: step >= start_step + W

    When opening, freeze the current link snapshot.
    """
    if W <= 0:
        raise ValueError(f"W must be positive, got {W}")

    if window_state is None:
        omega_seed = int(rng.integers(0, 2**31 - 1))
        # return CommWindowState(W=W, start_step=step, omega_seed=omega_seed, frozen_link=_clone_link(link_current))
        link_used = _orient_and_dedup_link(link_current, flags)
        return CommWindowState(W=W, start_step=step, omega_seed=omega_seed, frozen_link=_clone_link(link_used))
    
    if step >= window_state.start_step + window_state.W:
        omega_seed = int(rng.integers(0, 2**31 - 1))
        # return CommWindowState(W=W, start_step=step, omega_seed=omega_seed, frozen_link=_clone_link(link_current))
        link_used = _orient_and_dedup_link(link_current, flags)
        return CommWindowState(W=W, start_step=step, omega_seed=omega_seed, frozen_link=_clone_link(link_used))


    return window_state


def select_window_link(
    step: int,
    link_current: LinkSnapshot,
    window_state: Optional[CommWindowState],
    flags: FeatureFlags,
) -> LinkSnapshot:
    """
    If enable_link_freeze:
      return frozen_link (must exist)
    else:
      return link_current
    """
    if bool(getattr(flags, "enable_link_freeze", False)):
        if window_state is None:
            raise ValueError("enable_link_freeze=True but window_state is None")
        # return window_state.frozen_link
    # return link_current
        return _orient_and_dedup_link(window_state.frozen_link, flags)
    return _orient_and_dedup_link(link_current, flags)


def apply_ttl_filter(
    link: LinkSnapshot,
    ttl_hops: int,
    ttl_steps: int,
    flags: FeatureFlags,
    *,
    step: int | None = None,
    window_state: CommWindowState | None = None,
    N: int | None = None,
    params: object | None = None
) -> LinkSnapshot:
    """
    TTL filtering: in Step5 we assume upstream has already computed `link.is_stale`.
    - If enable_ttl_filter: remove stale edges (is_stale == True)
    - Else: pass through.

    NOTE:
    - ttl_hops/ttl_steps are kept for future extensions (e.g., compute staleness here),
      but current fallback uses only `is_stale` to avoid env dependency.
    """
    _ = ttl_hops
    enable_filter = bool(getattr(flags, "enable_ttl_filter", False))
    enable_engine = bool(getattr(flags, "enable_staleness_engine", True))

    if enable_engine:
        if step is None or window_state is None or N is None:
            raise ValueError("enable_stalness_engine = True requires step/window_state/N")
        update_last_seen(int(step), link_current=link, window_state=window_state, flags=flags)
    
        use_freeze = bool(getattr(flags, "enable_link_freeze", False))
        link_used = window_state.frozen_link if (use_freeze and window_state is not None) else link

        strict = bool(getattr(window_state, "staleness_strict", False))
        is_stale = compute_is_stale(int(step), link_used, window_state, flags, int(ttl_steps), strict=strict)
        active_hint = compute_active_hint(
            int(N),
            link_used,
            is_stale,
            step=int(step),
            window_state=window_state,
            flags=flags,
            params=params
        )
        attach_staleness_to_window(window_state=window_state, is_stale=is_stale, active_hint=active_hint)

        link_used2 = LinkSnapshot(
            robot_ids=list(link_used.robot_ids),
            edges = np.asarray(link_used.edges).copy(),
            signal=np.asarray(link_used.signal).copy(),
            capacity=np.asarray(link_used.capacity).copy(),
            delay=np.asarray(link_used.delay).copy(),
            plr=np.asarray(link_used.plr).copy(),
            is_stale=np.asarray(is_stale, dtype=bool).copy()
        )

        if not enable_filter:
            return link_used2
        
        stale = np.asarray(is_stale, dtype=bool).reshape(-1)
        if stale.shape[0] == 0:
            return link_used2
        keep = ~stale
        if keep.all():
            return link_used2
        
        return LinkSnapshot(
            robot_ids=list(link_used2.robot_ids),
            edges = np.asarray(link_used2.edges)[keep].copy(),
            signal=np.asarray(link_used2.signal)[keep].copy(),
            capacity=np.asarray(link_used2.capacity)[keep].copy(),
            delay=np.asarray(link_used2.delay)[keep].copy(),
            plr=np.asarray(link_used2.plr)[keep].copy(),
            is_stale=np.asarray(link_used2.is_stale)[keep].copy(),
        )
    
    use_freeze = bool(getattr(flags, "enable_link_freeze", False))
    link_used = window_state.frozen_link if (use_freeze and window_state is not None) else link

    if not enable_filter:
        return link_used
    
    stale = np.asarray(link_used.is_stale, dtype=bool).reshape(-1)
    if stale.shape[0] == 0:
        return link_used
    keep = ~stale
    if keep.all():
        return link_used
    
    return LinkSnapshot(
        robot_ids=list(link_used.robot_ids),
        edges = np.asarray(link_used.edges)[keep].copy(),
        signal=np.asarray(link_used.signal)[keep].copy(),
        capacity=np.asarray(link_used.capacity)[keep].copy(),
        delay=np.asarray(link_used.delay)[keep].copy(),
        plr=np.asarray(link_used.plr)[keep].copy(),
        is_stale=np.asarray(link_used.is_stale)[keep].copy(),
    )
