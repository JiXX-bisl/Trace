# scripts/tests/test_q_step_stageA.py
from __future__ import annotations

import numpy as np

from scripts.core.data import (
    CadmmProblem,
    CadmmParams,
    CadmmWarmStart,
    FeatureFlags,
    LinkSnapshot,
    TaskSnapshot,
    CommWindowState,
)
from scripts.core.inner.blocks import make_registry, decode_pos, set_block
from scripts.core.inner.q_step import solve_local_q
from scripts.core.cadmm_solver import solve_inner_cadmm


def _make_params(**overrides) -> CadmmParams:
    """Create a fully-specified CadmmParams (strict dataclass)."""
    base = dict(
        # solver
        eta=1.0,
        alpha=1.6,
        eps_pri=1e-9,
        eps_dual=1e-9,
        k_max=1,
        ttl_hops=1,
        t_fresh=1,
        mu=10.0,
        tau_incr=2.0,
        tau_decr=2.0,
        # costs
        move_w=0.0,
        cover_gamma=0.0,
        new_w=0.0,
        fe_w=0.0,
        role_task_weight=0.0,
        role_relay_weight=0.0,
        role_main_weight=0.0,
        qos_exec_weight=0.0,
        qos_relay_weight=0.0,
        qos_default_weight=0.0,
        qos_break_w=0.0,
        qos_improve_w=0.0,
        qos_degrade_w=0.0,
        urgency_w=0.0,
        rep_w=0.0,
        rep_sigma=1.0,
    )
    base.update(overrides)
    return CadmmParams(**base)


def _pos_only_flags() -> FeatureFlags:
    """Disable all blocks except pos, keep solver stable for E=0/G=0 tests."""
    flags = FeatureFlags()

    # blocks
    flags.enable_pos = True
    flags.enable_cov = False
    flags.enable_f_hat = False
    flags.enable_B_hat = False
    flags.enable_y_hat = False
    flags.enable_sigma = False
    flags.enable_r_hat = False

    # projections (only pos projection relevant)
    flags.enable_proj_pos = True
    flags.enable_proj_f_hat = False
    flags.enable_proj_B_hat = False
    flags.enable_proj_y_hat = False
    flags.enable_proj_sigma = False
    flags.enable_proj_r_hat = False

    # numeric / diagnostics
    flags.enable_residual_balancing = False
    flags.enable_over_relax = True

    # Stage A q-step switches (use setattr for backward compatibility during transition)
    setattr(flags, "enable_qstep_admm_term", True)
    setattr(flags, "enable_qstep_cost_move", False)
    setattr(flags, "enable_qstep_cost_explore", False)
    setattr(flags, "enable_qstep_cost_task", False)
    setattr(flags, "enable_qstep_cost_qos", False)
    setattr(flags, "enable_qstep_cost_repulsion", False)

    return flags


def _empty_link(N: int) -> LinkSnapshot:
    return LinkSnapshot(
        robot_ids=list(range(N)),
        edges=np.zeros((0, 2), dtype=np.int32),
        signal=np.zeros((0,), dtype=np.float32),
        capacity=np.zeros((0,), dtype=np.float32),
        delay=np.zeros((0,), dtype=np.float32),
        plr=np.zeros((0,), dtype=np.float32),
        is_stale=np.zeros((0,), dtype=bool),
    )


def _empty_window(link: LinkSnapshot) -> CommWindowState:
    return CommWindowState(W=1, start_step=0, omega_seed=0, frozen_link=link)


def _empty_task() -> TaskSnapshot:
    return TaskSnapshot(
        task_pos=np.zeros((0, 2), dtype=np.float32),
        deadline=np.zeros((0,), dtype=np.float32),
        priority=np.zeros((0,), dtype=np.float32),
        cluster_id=np.zeros((0,), dtype=np.int32),
    )


def _make_problem(
    *,
    N: int,
    candidates_by_robot: list[np.ndarray],
    robot_pos: np.ndarray,
    frontier_entropy: np.ndarray | None = None,
    flags: FeatureFlags | None = None,
    params: CadmmParams | None = None,
    task: TaskSnapshot | None = None,
) -> CadmmProblem:
    """Strict CadmmProblem construction (no SimpleNamespace)."""
    if flags is None:
        flags = _pos_only_flags()
    if params is None:
        params = _make_params()
    if task is None:
        task = _empty_task()

    link = _empty_link(N)
    window = _empty_window(link)

    if frontier_entropy is None:
        frontier_entropy = np.zeros((1, 1), dtype=np.float32)

    repulsion_grad = np.zeros((N, 2), dtype=np.float32)

    return CadmmProblem(
        N=N,
        E=0,
        G=0,
        T=int(np.asarray(task.task_pos).reshape(-1, 2).shape[0]),
        robot_pos=np.asarray(robot_pos, dtype=np.float32).reshape(N, 2),
        candidate_moves=[np.asarray(c, dtype=np.float32).reshape(-1, 2) for c in candidates_by_robot],
        coverage=float(0.0),
        frontier_entropy=np.asarray(frontier_entropy, dtype=np.float32),
        repulsion_grad=repulsion_grad,
        link=link,
        task=task,
        params=params,
        flags=flags,
        window=window,
    )


def test_q_step_move_cost_toggle_changes_choice():
    rng = np.random.default_rng(0)
    flags = _pos_only_flags()
    params = _make_params(move_w=100.0, eta=1.0)

    N = 1
    reg = make_registry(N=N, E=0, G=0, flags=flags)

    candidates = [np.asarray([[0, 0], [10, 0]], dtype=np.float32)]
    robot_pos = np.asarray([[0, 0]], dtype=np.float32)
    prob = _make_problem(N=N, candidates_by_robot=candidates, robot_pos=robot_pos, flags=flags, params=params)

    z = np.zeros((reg.total_dim,), dtype=np.float32)
    # set z target to [10,0]
    set_block(reg, z, "pos", np.asarray([10.0, 0.0], dtype=np.float32))
    u0 = np.zeros_like(z)

    # move OFF -> pure ADMM -> choose [10,0]
    setattr(flags, "enable_qstep_cost_move", False)
    q = solve_local_q(rid=0, problem=prob, reg=reg, z=z, u_i=u0, q_i_prev=z, rng=rng)
    chosen = decode_pos(reg, q, N=1)[0]
    assert np.allclose(chosen, [10.0, 0.0])

    # move ON & move_w huge -> choose [0,0]
    setattr(flags, "enable_qstep_cost_move", True)
    q2 = solve_local_q(rid=0, problem=prob, reg=reg, z=z, u_i=u0, q_i_prev=z, rng=rng)
    chosen2 = decode_pos(reg, q2, N=1)[0]
    assert np.allclose(chosen2, [0.0, 0.0])


def test_q_step_explore_cost_changes_choice():
    rng = np.random.default_rng(0)
    flags = _pos_only_flags()
    params = _make_params(new_w=1.0, eta=1.0)

    N = 1
    reg = make_registry(N=N, E=0, G=0, flags=flags)

    candidates = [np.asarray([[0, 0], [10, 0]], dtype=np.float32)]
    robot_pos = np.asarray([[0, 0]], dtype=np.float32)

    frontier_entropy = np.zeros((1, 11), dtype=np.float32)
    frontier_entropy[0, 10] = 1000.0  # (x=10,y=0) very attractive

    prob = _make_problem(
        N=N,
        candidates_by_robot=candidates,
        robot_pos=robot_pos,
        frontier_entropy=frontier_entropy,
        flags=flags,
        params=params,
    )

    z = np.zeros((reg.total_dim,), dtype=np.float32)
    # ADMM target = [0,0] (would prefer [0,0] if explore disabled)
    set_block(reg, z, "pos", np.asarray([0.0, 0.0], dtype=np.float32))
    u0 = np.zeros_like(z)

    setattr(flags, "enable_qstep_cost_explore", True)
    q = solve_local_q(rid=0, problem=prob, reg=reg, z=z, u_i=u0, q_i_prev=z, rng=rng)
    chosen = decode_pos(reg, q, N=1)[0]
    assert np.allclose(chosen, [10.0, 0.0])


def test_async_inactive_rows_unchanged_end_to_end():
    rng = np.random.default_rng(0)
    flags = _pos_only_flags()
    params = _make_params(k_max=1, eta=1.0)

    # async updates: round_robin with async_k=1 -> iter0 active robot = 0
    flags.enable_async_updates = True
    setattr(flags, "async_mode", "round_robin")
    setattr(flags, "async_k", 1)

    N = 4
    reg = make_registry(N=N, E=0, G=0, flags=flags)

    candidates = [np.asarray([[0, 0], [10, 0]], dtype=np.float32) for _ in range(N)]
    robot_pos = np.zeros((N, 2), dtype=np.float32)
    prob = _make_problem(N=N, candidates_by_robot=candidates, robot_pos=robot_pos, flags=flags, params=params)

    # warm start: z all zeros, q rows all equal to z, u all zeros except robot0 shifts target
    z = np.zeros((reg.total_dim,), dtype=np.float32)
    q0 = np.tile(z[None, :], (N, 1)).astype(np.float32)
    u0 = np.zeros((N, reg.total_dim), dtype=np.float32)

    # make u[0].pos = [-10,0] so target = (z - u[0]).pos = [10,0]
    pos_u = np.zeros((N, 2), dtype=np.float32)
    pos_u[0] = np.asarray([-10.0, 0.0], dtype=np.float32)
    set_block(reg, u0[0], "pos", pos_u.reshape(-1))

    ws = CadmmWarmStart(z=z.copy(), q=q0.copy(), u=u0.copy())

    sol, ws2, diag = solve_inner_cadmm(prob, warm_start=ws, rng=rng)
    _ = (ws2, diag)

    # inactive rows unchanged
    assert np.array_equal(sol.q[1:], ws.q[1:])
    # active row changed
    assert not np.array_equal(sol.q[0], ws.q[0])


def test_stageA_smoke_end_to_end_shapes_unchanged():
    rng = np.random.default_rng(123)
    flags = _pos_only_flags()
    params = _make_params(k_max=1, move_w=0.5, new_w=0.2, eta=1.0)

    setattr(flags, "enable_qstep_cost_move", True)
    setattr(flags, "enable_qstep_cost_explore", True)

    N = 2
    reg = make_registry(N=N, E=0, G=0, flags=flags)

    candidates = [np.asarray([[0, 0], [5, 0], [0, 5]], dtype=np.float32) for _ in range(N)]
    robot_pos = np.asarray([[0, 0], [1, 1]], dtype=np.float32)

    frontier_entropy = np.zeros((6, 6), dtype=np.float32)
    frontier_entropy[0, 5] = 10.0

    prob = _make_problem(
        N=N,
        candidates_by_robot=candidates,
        robot_pos=robot_pos,
        frontier_entropy=frontier_entropy,
        flags=flags,
        params=params,
    )

    sol, ws, diag = solve_inner_cadmm(prob, warm_start=None, rng=rng)

    assert sol.next_pos.shape == (N, 2)
    assert sol.z.ndim == 1
    assert sol.q.shape == (N, sol.z.shape[0])
    assert sol.u.shape == (N, sol.z.shape[0])
    assert ws.z.shape == sol.z.shape
    assert ws.q.shape == sol.q.shape
    assert ws.u.shape == sol.u.shape
    assert diag.iters <= params.k_max
