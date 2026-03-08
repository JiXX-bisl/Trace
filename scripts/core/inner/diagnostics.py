# scripts/core/inner/diagnostics.py
from __future__ import annotations

from typing import Tuple

import numpy as np

from scripts.core.data import CadmmProblem, CadmmDiagnostics, CommWindowState, FeatureFlags
from scripts.core.inner.blocks import BlockRegistry


def _orient_edge(i: int, j: int, root: int, mode: str) -> Tuple[int, int]:
    m = str(mode or "min_id")
    if m == "as_is":
        return int(i), int(j)
    if m == "root":
        if i == root and j != root:
            return int(i), int(j)
        if j == root and i != root:
            return int(j), int(i)
        a = int(min(i, j)); b = int(max(i, j))
        return a, b
    a = int(min(i, j)); b = int(max(i, j))
    return a, b


def _edge_key(i: int, j: int, window_state: CommWindowState, flags: FeatureFlags) -> Tuple[int, int]:
    # When we orient+dedup edges, cache keys must be directed (src,dst) to match edges order.
    orient = bool(getattr(flags, "assembled_orient_undirected_edges", False))
    if not orient:
        return (i, j) if i <= j else (j, i)
    # root id: prefer reachability root if available (set by staleness engine), else assembled_root_id
    root = int(getattr(window_state, "reachability_root_id", getattr(flags, "assembled_root_id", 0)))
    mode = str(getattr(flags, "assembled_edge_orientation_mode", "min_id"))
    src, dst = _orient_edge(int(i), int(j), root=root, mode=mode)
    return (src, dst)


def attach_stagec_diagnostics(problem: CadmmProblem, reg: BlockRegistry, diag: CadmmDiagnostics) -> None:
    """Attach Stage C optional runtime info to diagnostics via setattr."""
    _ = reg

    res_type = "linear" if bool(getattr(problem.flags, "use_linear_residual", False)) else "consensus"
    setattr(diag, "residual_model_type", res_type)

    E_total = int(np.asarray(problem.link.edges).reshape(-1, 2).shape[0])
    E_stale = int(np.asarray(getattr(problem.link, "is_stale", np.zeros((0,), dtype=bool)), dtype=bool).sum())

    window = getattr(problem, "window", None)
    hint = getattr(window, "active_hint", None) if window is not None else None
    N_active_hint = int(np.asarray(hint, dtype=bool).sum()) if hint is not None else 0

    setattr(diag, "staleness_stats", {"E_total": E_total, "E_stale": E_stale, "N_active_hint": N_active_hint})

    # has_ref = hasattr(problem.window, "ref_rate")
    has_ref = bool(window is not None and hasattr(window, "ref_rate"))
    setattr(diag, "stage0_ref_rate_present", bool(has_ref))
    if has_ref:
        # ref = np.asarray(getattr(problem.window, "ref_rate"), dtype=np.float32).reshape(-1)
        ref = np.asarray(getattr(window, "ref_rate"), dtype=np.float32).reshape(-1)
        if ref.size:
            summary = {"min": float(ref.min()), "max": float(ref.max()), "mean": float(ref.mean())}
        else:
            summary = {"min": 0.0, "max": 0.0, "mean": 0.0}
        setattr(diag, "ref_rate_summary", summary)
    else:
        setattr(diag, "ref_rate_summary", {"min": 0.0, "max": 0.0, "mean": 0.0})
    
    # budget cache / violation summaries
    has_cache = bool(window is not None and hasattr(window, "budget_cache"))
    setattr(diag, "budget_cache_present", bool(has_cache))
    if has_cache:
        cache_obj = getattr(window, "budget_cache")
        if isinstance(cache_obj, dict):
            vals = np.asarray(list(cache_obj.values()), dtype=np.float32).reshape(-1)
        else:
            vals = np.asarray(cache_obj, dtype=np.float32).reshape(-1)
        if vals.size:
            setattr(diag, "budget_cache_summary", {"min": float(vals.min()), "max": float(vals.max()), "mean": float(vals.mean())})
        else:
            setattr(diag, "budget_cache_summary", {"min": 0.0, "max": 0.0, "mean": 0.0})
    else:
        setattr(diag, "budget_cache_summary", {"min": 0.0, "max": 0.0, "mean": 0.0})
    
    viol = getattr(window, "budget_violation", 0.0) if window is not None else 0.0
    setattr(diag, "budget_violation", float(viol) if np.isfinite(float(viol)) else 0.0)
    
    if bool(getattr(problem.flags, "enable_budget_normalization", False)) and has_cache:
        from scripts.core.inner.qos_metrics import normalize_budget

        cache_obj = getattr(window, "budget_cache")
        if isinstance(cache_obj, dict):
            # Follow frozen edge ordering.
            edges = np.asarray(getattr(getattr(window, "frozen_link", None), "edges", np.zeros((0, 2), np.int32)), dtype=np.int32).reshape(-1, 2)


            cache = np.asarray([float(cache_obj.get(_edge_key(int(i), int(j), window, problem.flags), 0.0)) for (i, j) in edges], dtype=np.float32)
        else:
            cache = np.asarray(cache_obj, dtype=np.float32).reshape(-1)
        mode = str(getattr(problem.params, "budget_norm_mode", "per_edge_ref"))
        norm = normalize_budget(cache, window, mode=mode)
        if norm.size:
            setattr(
                diag,
                "budget_norm_summary",
                {"min": float(norm.min()), "max": float(norm.max()), "mean": float(norm.mean())},
            )
        else:
            setattr(diag, "budget_norm_summary", {"min": 0.0, "max": 0.0, "mean": 0.0})
    else:
        setattr(diag, "budget_norm_summary", {"min": 0.0, "max": 0.0, "mean": 0.0})

    # Record toggles
    setattr(diag, "budget_normalization_enabled", bool(getattr(problem.flags, "enable_budget_normalization", False)))
    setattr(diag, "budget_soft_violation_enabled", bool(getattr(problem.flags, "enable_budget_soft_violation", False)))

