# scripts/core/inner/stage0_init.py
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from scripts.core.data import CommWindowState, LinkSnapshot, CadmmParams, FeatureFlags


def _edge_key(i: int, j: int) -> Tuple[int, int]:
    return (i, j) if i <= j else (j, i)


def stage0_init_if_needed(
    step: int,
    window_state: CommWindowState,
    link_frozen: LinkSnapshot,
    params: CadmmParams,
    flags: FeatureFlags,
) -> None:
    """Stage-0 initialization at the start of a communication window.

    Implements the engineering mechanism without changing dataclasses.
    Attaches to window_state via setattr:
      - ref_rate: np.ndarray(E,) float32
      - budget_cache: dict(edge_key -> float)
      - stage0_inited_step: int
    """
    _ = flags

    if int(getattr(window_state, "start_step", step)) != int(step):
        return
    if int(getattr(window_state, "stage0_inited_step", -1)) == int(step):
        return

    mode = str(getattr(params, "stage0_ref_rate_mode", "capacity"))
    ratio = float(getattr(params, "stage0_budget_ratio", 1.0))
    ratio = max(ratio, 0.0)

    cap = np.asarray(getattr(link_frozen, "capacity", np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    if mode == "capacity":
        ref = cap
    else:
        # fallback: still use capacity
        ref = cap
    ref_rate = (ref * ratio).astype(np.float32, copy=False)
    setattr(window_state, "ref_rate", ref_rate)

    edges = np.asarray(link_frozen.edges, dtype=np.int32).reshape(-1, 2)
    budget_cache: Dict[Tuple[int, int], float] = {}
    for (i, j) in edges:
        budget_cache[_edge_key(int(i), int(j))] = 0.0
    setattr(window_state, "budget_cache", budget_cache)

    setattr(window_state, "stage0_inited_step", int(step))
    # TODO: ref_rate can be replaced by Lee model / running statistics.
    # extra summaries
    try:
        setattr(window_state, "ref_total", float(np.sum(ref_rate)))
        if ref_rate.size > 0:
            setattr(
                window_state,
                "ref_rate_summary",
                {
                    "min": float(np.min(ref_rate)),
                    "max": float(np.max(ref_rate)),
                    "mean": float(np.mean(ref_rate))
                }
            )
        else:
            setattr(window_state, "ref_rate_summary", {"min": 0.0, "max": 0.0, "mean": 0.0})
    except Exception:
        pass
