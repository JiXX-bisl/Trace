# scripts/core/inner/async_scheduler.py
from __future__ import annotations

import numpy as np
from scripts.core.data import FeatureFlags


def _get_update_count(N: int, flags: FeatureFlags) -> int:
    """Derive m (number of active robots) from flags.async_k or flags.async_ratio."""
    m = int(getattr(flags, "async_k", 0) or 0)
    if m > 0:
        return min(N, m)

    ratio = float(getattr(flags, "async_ratio", 1.0))
    ratio = max(0.0, min(1.0, ratio))
    if ratio >= 1.0:
        return N
    return max(1, int(round(ratio * N)))


def get_active_set(
    iter_k: int,
    N: int,
    flags: FeatureFlags,
    rng: np.random.Generator,
    *,
    active_hint: np.ndarray | None = None
) -> np.ndarray:
    """
    Return active mask of shape (N,) bool.

    Modes (flags.async_mode):
      - "all"         : all active (default / baseline)
      - "round_robin" : rotate active subset deterministically
      - "random_k"    : randomly sample m robots each iter

    If enable_async_updates is False -> all active.
    """
    if not bool(getattr(flags, "enable_async_updates", False)):
        return np.ones((N,), dtype=bool)

    mode = str(getattr(flags, "async_mode", "all"))
    if mode == "all":
        return np.ones((N,), dtype=bool)
    
    if mode == "ttl_freshness":
        if active_hint is not None:
            hint = np.asarray(active_hint, dtype=bool).reshape(-1)
            if hint.shape == (N,) and bool(np.any(hint)):
                return hint.copy()
        mode = "round_robin"

    m = _get_update_count(N, flags)
    mask = np.zeros((N,), dtype=bool)

    if mode == "round_robin":
        # contiguous chunk rotated by iter_k
        start = (iter_k * m) % N
        idx = [(start + t) % N for t in range(m)]
        mask[idx] = True
        return mask

    if mode == "random_k":
        idx = rng.choice(N, size=m, replace=False)
        mask[idx] = True
        return mask

    raise ValueError(f"Unknown async_mode='{mode}'. Expected 'all'/'round_robin'/'random_k'.")
