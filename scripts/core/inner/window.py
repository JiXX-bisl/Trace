# scripts/core/inner/window.py
from __future__ import annotations

from typing import Optional
import numpy as np

from scripts.core.data import LinkSnapshot, CommWindowState, FeatureFlags


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
        return CommWindowState(W=W, start_step=step, omega_seed=omega_seed, frozen_link=_clone_link(link_current))

    if step >= window_state.start_step + window_state.W:
        omega_seed = int(rng.integers(0, 2**31 - 1))
        return CommWindowState(W=W, start_step=step, omega_seed=omega_seed, frozen_link=_clone_link(link_current))

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
        return window_state.frozen_link
    return link_current


def apply_ttl_filter(
    link: LinkSnapshot,
    ttl_hops: int,
    ttl_steps: int,
    flags: FeatureFlags,
) -> LinkSnapshot:
    """
    TTL filtering: in Step5 we assume upstream has already computed `link.is_stale`.
    - If enable_ttl_filter: remove stale edges (is_stale == True)
    - Else: pass through.

    NOTE:
    - ttl_hops/ttl_steps are kept for future extensions (e.g., compute staleness here),
      but current fallback uses only `is_stale` to avoid env dependency.
    """
    if not bool(getattr(flags, "enable_ttl_filter", False)):
        return link

    stale = np.asarray(link.is_stale, dtype=bool).reshape(-1)
    if stale.shape[0] == 0:
        return link
    keep = ~stale
    if keep.all():
        return link

    return LinkSnapshot(
        robot_ids=list(link.robot_ids),
        edges=np.asarray(link.edges)[keep].copy(),
        signal=np.asarray(link.signal)[keep].copy(),
        capacity=np.asarray(link.capacity)[keep].copy(),
        delay=np.asarray(link.delay)[keep].copy(),
        plr=np.asarray(link.plr)[keep].copy(),
        is_stale=np.asarray(link.is_stale)[keep].copy(),
    )
