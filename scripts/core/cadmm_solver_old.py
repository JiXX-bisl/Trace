from __future__ import annotations

import numpy as np
import pytest

from scripts.core.data import (
    CadmmProblem,
    CadmmWarmStart,
    FeatureFlags,
    LinkSnapshot,
    TaskSnapshot,
    CommWindowState,
)
from scripts.core.inner.blocks import make_registry
from scripts.core.inner.constraints import build_constraints
from scripts.core.inner.projections import project_blocks
from scripts.core.cadmm_solver import solve_inner_cadmm


class _Params:
    """Minimal params stub with required names in your CadmmParams."""
    def __init__(self):
        self.eta = 1.0
        self.alpha = 1.6
        self.eps_pri = 1e-4
        self.eps_dual = 1e-4
        self.k_max = 4
        self.ttl_hops = 1
        self.t_fresh = 1
        self.mu = 10.0
        self.tau_incr = 2.0
        self.tau_decr = 2.0

        # cost weights (unused in minimal solver but exist in your CadmmParams)
        self.move_w = 1.0
        self.cover_gamma = 1.0
        self.new_w = 1.0
        self.fe_w = 1.0
        self.role_task_weight = 1.0
        self.role_relay_weight = 1.0
        self.role_main_weight = 1.0
        self.qos_exec_weight = 1.0
        self.qos_relay_weight = 1.0
        self.qos_default_weight = 1.0
        self.qos_break_w = 1.0
        self.qos_improve_w = 1.0
        self.qos_degrade_w = 1.0
        self.urgency_w = 1.0
        self.rep_w = 1.0
        self.rep_sigma = 1.0


def _make_problem(flags: FeatureFlags, params: _Params) -> CadmmProblem:
    N, E, G, T = 3, 2, 2, 2

    robot_pos = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)

    candidate_moves = []
    for i in range(N):
        base = robot_pos[i]
        moves = np.stack(
            [base,
             base + [1, 0],
             base + [0, 1],
             base + [-1, 0],
             base + [0, -1]],
            axis=0
        ).astype(np.float32)
        candidate_moves.append(moves)

    # link snapshot
    robot_ids = [0, 1, 2]
    edges = np.array([[0, 1], [1, 2]], dtype=np.int32)
    signal = np.array([-50.0, -60.0], dtype=np.float32)
    capacity = np.array([3.0, 5.0], dtype=np.float32)
    delay = np.array([0.1, 0.2], dtype=np.float32)
    plr = np.array([0.01, 0.02], dtype=np.float32)
    is_stale = np.array([False, False], dtype=bool)
    link = LinkSnapshot(robot_ids=robot_ids, edges=edges, signal=signal, capacity=capacity, delay=delay, plr=plr, is_stale=is_stale)

    # task snapshot
    task_pos = np.array([[5.0, 5.0], [6.0, 6.0]], dtype=np.float32)
    deadline = np.array([10.0, 20.0], dtype=np.float32)
    priority = np.array([1.0, 2.0], dtype=np.float32)
    cluster_id = np.array([0, 1], dtype=np.int32)
    task = TaskSnapshot(task_pos=task_pos, deadline=deadline, priority=priority, cluster_id=cluster_id)

    # window state
    window = CommWindowState(W=5, start_step=0, omega_seed=0, frozen_link=link)

    frontier_entropy = np.zeros((8, 8), dtype=np.float32)
    repulsion_grad = np.zeros((N, 2), dtype=np.float32)

    return CadmmProblem(
        N=N, E=E, G=G, T=T,
        robot_pos=robot_pos,
        candidate_moves=candidate_moves,
        coverage=0.0,
        frontier_entropy=frontier_entropy,
        repulsion_grad=repulsion_grad,
        link=link,
        task=task,
        params=params,
        flags=flags,
        window=window,
    )


def test_inner_cadmm_runs_shapes_and_diag_fields():
    rng = np.random.default_rng(0)
    flags = FeatureFlags()
    # make projection deterministic-ish for tests
    setattr(flags, "proj_method", "dykstra")
    setattr(flags, "proj_iters", 80)
    setattr(flags, "proj_tol", 1e-8)
    setattr(flags, "enable_over_relax", False)

    params = _Params()
    problem = _make_problem(flags, params)

    sol, ws, diag = solve_inner_cadmm(problem, warm_start=None, rng=rng)

    # registry computed inside solver
    reg = make_registry(problem.N, problem.E, problem.G, problem.flags)
    D = reg.total_dim

    assert sol.next_pos.shape == (problem.N, 2)
    assert sol.z.shape == (D,)
    assert sol.q.shape == (problem.N, D)
    assert sol.u.shape == (problem.N, D)

    assert ws.z.shape == (D,)
    assert ws.q.shape == (problem.N, D)
    assert ws.u.shape == (problem.N, D)

    assert isinstance(diag.iters, int) and 1 <= diag.iters <= params.k_max
    assert isinstance(diag.r_norm, dict)
    assert isinstance(diag.s_norm, dict)
    assert isinstance(diag.eta_hist, list) and len(diag.eta_hist) == diag.iters
    assert isinstance(diag.proj_violation, dict)


def test_projection_effect_end_to_end_violation_reduces():
    rng = np.random.default_rng(1)
    flags = FeatureFlags()
    setattr(flags, "proj_method", "dykstra")
    setattr(flags, "proj_iters", 120)
    setattr(flags, "proj_tol", 1e-9)
    setattr(flags, "enable_over_relax", False)

    params = _Params()
    params.k_max = 1  # single iteration should call projection once
    problem = _make_problem(flags, params)

    reg = make_registry(problem.N, problem.E, problem.G, problem.flags)
    D = reg.total_dim

    # craft violating warmstart z:
    z0 = np.zeros((D,), dtype=np.float32)
    z_blocks = reg.unpack(z0)

    if "f_hat" in z_blocks:
        z_blocks["f_hat"] = np.array([10.0, 10.0], dtype=np.float32)  # > capacity
    if "B_hat" in z_blocks:
        z_blocks["B_hat"] = np.array([10.0, 10.0], dtype=np.float32)  # > capacity
    if "y_hat" in z_blocks:
        z_blocks["y_hat"] = np.zeros((problem.G,), dtype=np.float32)   # violates simplex sum=1
    z0 = reg.pack(z_blocks)

    q0 = np.tile(z0[None, :], (problem.N, 1))
    u0 = np.zeros_like(q0)

    ws0 = CadmmWarmStart(z=z0, q=q0, u=u0)
    sol, ws, diag = solve_inner_cadmm(problem, warm_start=ws0, rng=rng)

    # verify projected z is feasible (in tolerance) for key constraints
    z_new_blocks = reg.unpack(sol.z)
    cons = build_constraints(problem, reg, z_blocks=z_new_blocks)

    # reprojection should not change much
    z_proj_blocks, _ = project_blocks(z_new_blocks, cons, method="dykstra", iters=120, tol=1e-10)
    z_reproj = reg.pack(z_proj_blocks)
    assert np.linalg.norm(sol.z - z_reproj) < 1e-5

    if "f_hat" in z_new_blocks:
        assert np.all(z_new_blocks["f_hat"] <= problem.link.capacity + 1e-6)
        assert np.all(z_new_blocks["f_hat"] >= -1e-6)

    if "B_hat" in z_new_blocks:
        assert np.all(z_new_blocks["B_hat"] <= problem.link.capacity + 1e-6)
        assert np.all(z_new_blocks["B_hat"] >= -1e-6)

    if "y_hat" in z_new_blocks:
        assert np.all(z_new_blocks["y_hat"] >= -1e-8)
        assert abs(float(np.sum(z_new_blocks["y_hat"])) - 1.0) < 1e-4


def test_warm_start_compatible_and_disable_block():
    rng = np.random.default_rng(2)

    flags = FeatureFlags()
    setattr(flags, "proj_method", "dykstra")
    setattr(flags, "proj_iters", 80)
    setattr(flags, "proj_tol", 1e-8)

    params = _Params()
    params.k_max = 3
    problem = _make_problem(flags, params)

    sol1, ws1, diag1 = solve_inner_cadmm(problem, warm_start=None, rng=rng)
    sol2, ws2, diag2 = solve_inner_cadmm(problem, warm_start=ws1, rng=rng)

    reg1 = make_registry(problem.N, problem.E, problem.G, problem.flags)
    assert sol1.z.shape == sol2.z.shape == (reg1.total_dim,)

    # disable a block (e.g., sigma) -> D shrinks and solver still runs
    flags2 = FeatureFlags(enable_sigma=False)
    setattr(flags2, "proj_method", "dykstra")
    setattr(flags2, "proj_iters", 80)
    setattr(flags2, "proj_tol", 1e-8)

    problem2 = _make_problem(flags2, params)
    reg2 = make_registry(problem2.N, problem2.E, problem2.G, problem2.flags)
    assert reg2.total_dim < reg1.total_dim

    sol3, ws3, diag3 = solve_inner_cadmm(problem2, warm_start=None, rng=rng)
    assert sol3.z.shape == (reg2.total_dim,)
    assert sol3.q.shape == (problem2.N, reg2.total_dim)
    assert sol3.u.shape == (problem2.N, reg2.total_dim)
