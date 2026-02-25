# scripts/tests/test_inner_cadmm_flag_matrix.py
from __future__ import annotations

import dataclasses
from dataclasses import replace
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from scripts.core.data import (
    CadmmParams,
    FeatureFlags,
    LinkSnapshot,
    TaskSnapshot,
    CommWindowState,
    CadmmProblem,
    CadmmWarmStart,
)
from scripts.core.cadmm_solver import solve_inner_cadmm_entry, build_problem_snapshot
from scripts.core.inner.blocks import make_registry


# -----------------------------------------------------------------------------
# 0) Deterministic dummy snapshot factory (pure numpy, no env dependency)
# -----------------------------------------------------------------------------


def _make_base_params(*, k_max: int = 2) -> CadmmParams:
    """
    Build a minimal CadmmParams instance that satisfies your dataclass constructor.
    Many algorithmic knobs are read via getattr(..., default) in the pipeline,
    so we only need to populate REQUIRED fields.
    """
    return CadmmParams(
        eta=1.0,
        alpha=1.0,        # over-relax safe baseline (alpha=1 => no effect)
        eps_pri=1e-2,     # loose thresholds to reduce iterations if convergence happens
        eps_dual=1e-2,
        k_max=int(k_max),
        ttl_hops=2,
        t_fresh=2,
        mu=10.0,
        tau_incr=2.0,
        tau_decr=2.0,
        move_w=0.05,
        cover_gamma=0.0,
        new_w=0.0,
        fe_w=0.0,
        role_task_weight=1.0,
        role_relay_weight=1.0,
        role_main_weight=1.0,
        qos_exec_weight=1.0,
        qos_relay_weight=1.0,
        qos_default_weight=1.0,
        qos_break_w=0.0,
        qos_improve_w=0.0,
        qos_degrade_w=0.0,
        urgency_w=0.1,
        rep_w=0.0,
        rep_sigma=1.0,
        # optional fields in your dataclass already have defaults:
        # staleness_strict/eps_rel/assembled_eps/root_id_default/ref_rate_*/...
        budget_cache_ema=0.5,
    )


def _make_link_snapshot(N: int, edges: np.ndarray) -> LinkSnapshot:
    edges = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
    E = int(edges.shape[0])
    # simple, stable QoS vectors
    cap = np.linspace(10.0, 5.0, num=max(E, 1), dtype=np.float32)[:E]
    dly = np.linspace(0.2, 1.0, num=max(E, 1), dtype=np.float32)[:E]
    plr = np.linspace(0.01, 0.20, num=max(E, 1), dtype=np.float32)[:E]
    sig = np.linspace(-40.0, -70.0, num=max(E, 1), dtype=np.float32)[:E]
    stale = np.zeros((E,), dtype=bool)
    return LinkSnapshot(
        robot_ids=list(range(int(N))),
        edges=edges,
        signal=sig,
        capacity=cap,
        delay=dly,
        plr=plr,
        is_stale=stale,
    )


def _make_task_snapshot(T: int, G: int, *, rng: np.random.Generator) -> TaskSnapshot:
    T = int(T)
    G = int(G)
    if T <= 0:
        return TaskSnapshot(
            task_pos=np.zeros((0, 2), dtype=np.float32),
            deadline=np.zeros((0,), dtype=np.float32),
            priority=np.zeros((0,), dtype=np.float32),
            cluster_id=np.zeros((0,), dtype=np.int32),
        )
    task_pos = rng.integers(low=0, high=20, size=(T, 2), dtype=np.int32).astype(np.float32)
    deadline = rng.integers(low=1, high=20, size=(T,), dtype=np.int32).astype(np.float32)
    priority = rng.random(size=(T,), dtype=np.float32) + 0.5
    cluster_id = rng.integers(low=0, high=max(G, 1), size=(T,), dtype=np.int32)
    return TaskSnapshot(task_pos=task_pos, deadline=deadline, priority=priority, cluster_id=cluster_id)


def _make_window_state(
    *,
    W: int,
    start_step: int,
    frozen_link: LinkSnapshot,
) -> CommWindowState:
    return CommWindowState(W=int(W), start_step=int(start_step), omega_seed=0, frozen_link=frozen_link)


def _make_env_problem(
    flags: FeatureFlags,
    params: CadmmParams,
    *,
    N: int = 4,
    G: int = 3,
    T: int = 4,
    stale_extra_edge_in_frozen: bool = False,
    rng: np.random.Generator | None = None,
) -> CadmmProblem:
    """
    Build a CadmmProblem that is valid for *most* flag combinations.

    If stale_extra_edge_in_frozen=True:
      - window.frozen_link has an extra edge that does NOT appear in current link,
        enabling staleness engine to mark it stale after ttl_steps.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    N = int(N)
    G = int(G)
    T = int(T)

    # robot pos in a small grid
    robot_pos = np.stack([np.array([2 + 3 * i, 2 + 2 * i], dtype=np.float32) for i in range(N)], axis=0)

    # candidates around current pos (include staying)
    deltas = np.array(
        [
            [0, 0],
            [1, 0],
            [-1, 0],
            [0, 1],
            [0, -1],
        ],
        dtype=np.float32,
    )
    candidate_moves: List[np.ndarray] = []
    for i in range(N):
        base = robot_pos[i][None, :]
        candidate_moves.append(base + deltas)

    frontier_entropy = rng.random((20, 20), dtype=np.float32)
    repulsion_grad = np.zeros((N, 2), dtype=np.float32)
    coverage = float(rng.random())

    # Current link edges (connected-ish)
    edges_cur = np.array([[0, 1], [1, 2], [2, 3], [0, 2], [1, 3]], dtype=np.int32)
    link_current = _make_link_snapshot(N, edges_cur)

    # Frozen link (may contain extra unseen edge)
    if stale_extra_edge_in_frozen:
        # add an edge not present in current edges -> never last_seen updated
        edges_frozen = np.vstack([edges_cur, np.array([[0, 3]], dtype=np.int32)])
    else:
        edges_frozen = edges_cur.copy()
    frozen_link = _make_link_snapshot(N, edges_frozen)

    window = _make_window_state(W=1000, start_step=0, frozen_link=frozen_link)
    task = _make_task_snapshot(T, G, rng=rng)

    # NOTE: CadmmProblem.E should match CURRENT link edges count
    return CadmmProblem(
        N=N,
        E=int(link_current.edges.shape[0]),
        G=G,
        T=T,
        robot_pos=robot_pos,
        candidate_moves=candidate_moves,
        coverage=coverage,
        frontier_entropy=frontier_entropy,
        repulsion_grad=repulsion_grad,
        link=link_current,
        task=task,
        params=params,
        flags=flags,
        window=window,
    )


# -----------------------------------------------------------------------------
# 1) Warm-start constructor (registry-aware, avoids "all-zero" corner cases)
# -----------------------------------------------------------------------------


def _make_registry_consistent_warm_start(problem_snapshot: CadmmProblem) -> CadmmWarmStart:
    """
    Construct a warm start that is:
      - dimension-consistent with make_registry(...)
      - "reasonable" (simplex y_hat, sigma derived, edge blocks within capacity)
    """
    reg = make_registry(problem_snapshot.N, problem_snapshot.E, problem_snapshot.G, problem_snapshot.flags)
    blocks: Dict[str, np.ndarray] = {}

    N = int(problem_snapshot.N)
    E = int(problem_snapshot.E)
    G = int(problem_snapshot.G)

    if "pos" in reg.names():
        blocks["pos"] = np.asarray(problem_snapshot.robot_pos, dtype=np.float32).reshape(-1)

    if "cov" in reg.names():
        blocks["cov"] = np.asarray([float(problem_snapshot.coverage)], dtype=np.float32)

    if "y_hat" in reg.names():
        if G > 0:
            blocks["y_hat"] = (np.ones((G,), dtype=np.float32) / float(G)).astype(np.float32)
        else:
            blocks["y_hat"] = np.zeros((0,), dtype=np.float32)

    if "sigma" in reg.names():
        # safe: keep within [0,1]
        if G > 0:
            blocks["sigma"] = (np.ones((G,), dtype=np.float32) / float(G)).astype(np.float32)
        else:
            blocks["sigma"] = np.zeros((0,), dtype=np.float32)

    if "B_hat" in reg.names():
        cap = np.asarray(getattr(problem_snapshot.link, "capacity", np.zeros((E,), np.float32)), dtype=np.float32).reshape(-1)
        if cap.size != E:
            cap = np.full((E,), 1.0, dtype=np.float32)
        blocks["B_hat"] = np.clip(0.5 * cap, 0.0, cap).astype(np.float32)

    if "f_hat" in reg.names():
        cap = np.asarray(getattr(problem_snapshot.link, "capacity", np.zeros((E,), np.float32)), dtype=np.float32).reshape(-1)
        if cap.size != E:
            cap = np.full((E,), 1.0, dtype=np.float32)
        blocks["f_hat"] = np.clip(0.25 * cap, 0.0, cap).astype(np.float32)

    if "r_hat" in reg.names():
        blocks["r_hat"] = np.zeros((N,), dtype=np.float32)

    z0 = reg.pack(blocks)
    q0 = np.tile(z0[None, :], (N, 1)).astype(np.float32, copy=False)
    u0 = np.zeros_like(q0, dtype=np.float32)
    return CadmmWarmStart(z=z0.astype(np.float32), q=q0, u=u0)


def _assert_outputs_finite_and_shaped(sol, ws, diag, *, N_expected: int) -> None:
    assert sol.next_pos.shape == (N_expected, 2)
    assert sol.z.ndim == 1
    D = int(sol.z.size)
    assert sol.q.shape == (N_expected, D)
    assert sol.u.shape == (N_expected, D)

    assert ws.z.shape == (D,)
    assert ws.q.shape == (N_expected, D)
    assert ws.u.shape == (N_expected, D)

    assert isinstance(diag.iters, int) and diag.iters >= 1
    assert isinstance(diag.r_norm, dict)
    assert isinstance(diag.s_norm, dict)
    assert isinstance(diag.proj_violation, dict)

    assert np.all(np.isfinite(sol.z))
    assert np.all(np.isfinite(sol.q))
    assert np.all(np.isfinite(sol.u))
    for v in diag.proj_violation.values():
        assert np.isfinite(float(v))


# -----------------------------------------------------------------------------
# 2) Case generator: all bool flags toggled + extra enum/string combos
# -----------------------------------------------------------------------------


BASE_FLAGS = FeatureFlags()
BASE_PARAMS = _make_base_params(k_max=2)


def _speed_patch_flags(flags: FeatureFlags) -> FeatureFlags:
    """
    Keep tests fast and stable:
      - reduce projection iterations
      - keep tolerances loose
    """
    # Only override if the field exists (it does in your dataclass)
    return replace(flags, proj_iters=min(int(getattr(flags, "proj_iters", 10)), 10), proj_tol=1e-4)


def _bool_toggle_cases(base: FeatureFlags) -> List[Tuple[str, Dict[str, Any]]]:
    cases: List[Tuple[str, Dict[str, Any]]] = []
    for f in dataclasses.fields(base):
        name = f.name
        val = getattr(base, name)
        if isinstance(val, bool):
            cases.append((f"toggle_{name}", {name: (not val)}))
    return cases


EXTRA_CASES: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = [
    # (case_name, flag_overrides, env_overrides)
    ("async_round_robin_k2", {"enable_async_updates": True, "async_mode": "round_robin", "async_k": 2}, {}),
    ("async_random_k2", {"enable_async_updates": True, "async_mode": "random_k", "async_k": 2}, {}),
    ("async_ttl_freshness", {"enable_async_updates": True, "async_mode": "ttl_freshness", "enable_ttl_filter": True}, {}),

    ("active_hint_reachability", {"enable_ttl_filter": True, "active_hint_mode": "reachability"}, {}),
    ("active_hint_reachability_memory", {"enable_ttl_filter": True, "active_hint_mode": "reachability_with_memory"}, {}),

    ("linear_residual_basic", {"use_linear_residual": True}, {}),
    ("linear_residual_with_diag_groups", {"use_linear_residual": True, "log_linear_groups": True}, {}),
    (
        "linear_residual_with_coupled_groups_safe",
        {
            "use_linear_residual": True,
            "enable_coupled_groups": True,
            "linear_assembly_mode": "coupled_basic",
            "include_coupled_groups_in_stop": True,
            # make sure y_hat/B_hat/f_hat are not all-zero so coupled residuals are well-defined
            "enable_qstep_update_y_hat": True,
            "enable_qstep_update_B_hat": True,
            "enable_qstep_update_f_hat": True,
        },
        {},
    ),

    (
        "assembled_ops_pos_only",
        {"enable_assembled_ops": True, "assembled_owner_mode": "pos_only"},
        {},
    ),
    (
        "assembled_ops_edge_by_incident",
        {"enable_assembled_ops": True, "assembled_owner_mode": "edge_by_incident"},
        {},
    ),
    (
        "assembled_ops_root_only",
        {"enable_assembled_ops": True, "assembled_owner_mode": "root_only"},
        {},
    ),

    (
        "budget_cache_closed_loop_q_source",
        {
            "enable_budget_cache_update": True,
            "enable_budget_soft_violation": True,
            "enable_budget_normalization": True,
            "budget_update_source": "q",
            "enable_qstep_update_B_hat": True,
        },
        {},
    ),

    (
        "flow_coupled_to_budget_constraints",
        {
            "enable_flow_coupled_to_budget": True,
            "enable_qstep_update_B_hat": True,
            "enable_qstep_update_f_hat": True,
        },
        {},
    ),

    (
        "proj_serial_method",
        {"proj_method": "serial", "proj_iters": 5},
        {},
    ),

    # TTL regression: frozen has an extra edge never "seen" in current -> stale after ttl
    (
        "ttl_freeze_stale_edge_drop",
        {"enable_link_freeze": True, "enable_ttl_filter": True, "enable_staleness_engine": True},
        {"stale_extra_edge_in_frozen": True, "step": 5},
    ),
]


# -----------------------------------------------------------------------------
# 3) Tests
# -----------------------------------------------------------------------------


def test_pipeline_cold_start_baseline_runs():
    """
    Cold-start sanity check for the whole entry pipeline.
    """
    flags = _speed_patch_flags(BASE_FLAGS)
    params = BASE_PARAMS

    env = _make_env_problem(flags, params, rng=np.random.default_rng(0))
    sol, ws, diag = solve_inner_cadmm_entry(
        env=env,
        step=0,
        params=params,
        flags=flags,
        warm_start=None,
        rng=np.random.default_rng(1),
        optional_overrides=None,
    )
    _assert_outputs_finite_and_shaped(sol, ws, diag, N_expected=env.N)


@pytest.mark.parametrize(
    "case_name,overrides",
    _bool_toggle_cases(BASE_FLAGS),
    ids=lambda x: x if isinstance(x, str) else None,
)
def test_pipeline_runs_for_each_bool_flag_toggle(case_name: str, overrides: Dict[str, Any]):
    """
    For every bool switch in FeatureFlags, toggle it (one-at-a-time) and ensure:
      - solve_inner_cadmm_entry runs end-to-end
      - outputs have consistent shapes and finite values
    Uses a registry-consistent warm start to avoid "all-zero" degenerate corners.
    """
    flags = replace(BASE_FLAGS, **overrides)
    flags = _speed_patch_flags(flags)
    params = BASE_PARAMS

    env = _make_env_problem(flags, params, rng=np.random.default_rng(0))

    # Build snapshot once to produce a dimension-consistent warm start.
    snap = build_problem_snapshot(
        env_or_state=env,
        step=0,
        params=params,
        flags=flags,
        rng=np.random.default_rng(123),
        optional_overrides=None,
    )
    warm = _make_registry_consistent_warm_start(snap)

    sol, ws, diag = solve_inner_cadmm_entry(
        env=env,
        step=0,
        params=params,
        flags=flags,
        warm_start=warm,
        rng=np.random.default_rng(1),
        optional_overrides=None,
    )
    _assert_outputs_finite_and_shaped(sol, ws, diag, N_expected=env.N)


@pytest.mark.parametrize(
    "case_name,flag_overrides,env_overrides",
    EXTRA_CASES,
    ids=[c[0] for c in EXTRA_CASES],
)
def test_pipeline_runs_for_extra_enum_and_combo_cases(
    case_name: str, flag_overrides: Dict[str, Any], env_overrides: Dict[str, Any]
):
    """
    Representative combinations for string/enum switches and multi-feature interactions.
    """
    flags = replace(BASE_FLAGS, **flag_overrides)
    flags = _speed_patch_flags(flags)
    params = BASE_PARAMS

    step = int(env_overrides.get("step", 0))
    stale_extra_edge = bool(env_overrides.get("stale_extra_edge_in_frozen", False))

    env = _make_env_problem(
        flags,
        params,
        stale_extra_edge_in_frozen=stale_extra_edge,
        rng=np.random.default_rng(0),
    )

    # snapshot -> warm start (dimension-safe)
    snap = build_problem_snapshot(
        env_or_state=env,
        step=step,
        params=params,
        flags=flags,
        rng=np.random.default_rng(123),
        optional_overrides=None,
    )
    warm = _make_registry_consistent_warm_start(snap)

    sol, ws, diag = solve_inner_cadmm_entry(
        env=env,
        step=step,
        params=params,
        flags=flags,
        warm_start=warm,
        rng=np.random.default_rng(1),
        optional_overrides=None,
    )
    _assert_outputs_finite_and_shaped(sol, ws, diag, N_expected=env.N)

    # Optional: when TTL is involved, check diag carries staleness stats (Stage C hook)
    if bool(getattr(flags, "enable_ttl_filter", False)):
        assert hasattr(diag, "staleness_stats")


def test_invalid_config_all_blocks_disabled_should_raise():
    """
    Registry cannot be empty. If user disables ALL macro blocks, solver should raise.
    """
    flags = replace(
        BASE_FLAGS,
        enable_pos=False,
        enable_cov=False,
        enable_f_hat=False,
        enable_B_hat=False,
        enable_y_hat=False,
        enable_sigma=False,
        enable_r_hat=False,
    )
    flags = _speed_patch_flags(flags)
    params = BASE_PARAMS
    env = _make_env_problem(flags, params, rng=np.random.default_rng(0))

    with pytest.raises(ValueError):
        _ = solve_inner_cadmm_entry(
            env=env,
            step=0,
            params=params,
            flags=flags,
            warm_start=None,
            rng=np.random.default_rng(1),
            optional_overrides=None,
        )