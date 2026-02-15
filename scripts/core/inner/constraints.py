"""
scripts.core.inner.constraints

Feasible-set *description* layer for TRACE / Inner C-ADMM.

This module **assembles** per-macro-block constraint description objects
(BoxC/SimplexC/AffineC/HalfspaceC/PolytopeC/...) that will later be consumed
by a *projection* layer.

Important boundaries (STRICT)
-----------------------------
1) This file does **NOT** compute projections.
2) It does **NOT** depend on env_state / robot classes.
3) It depends only on:
   - dataclasses defined in scripts.core.data (CadmmProblem + constraint objects)
   - BlockRegistry from scripts.core.blocks (only for slice/shape validation)

Unified I/O protocol
--------------------
Let D = reg.total_dim.
  z: (D,)
  q/u: (N, D)
Block access must go through reg.sl(name) / reg.shape(name) / reg.unpack(...).

This stage implements a *simplified* set of constraints (box/simplex/halfspace/affine)
to validate the constraint assembly pipeline and block dimension consistency.
Complex coupled constraints (e.g., Pi_P(sigma; y_hat)) are reserved as TODO
but the function hooks are kept for A/B migration.
"""

from __future__ import annotations

from dataclasses import is_dataclass, fields
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

import scripts.core.inner.coverage_metrics as coverage_metrics
from scripts.core.inner.blocks import BlockRegistry
from scripts.core.data import (
    CadmmProblem, 
    BoxC,
    SimplexC,
    HalfspaceC,
    AffineC,
    PolytopeC
)

# ----------------------------------
# Public API
# ----------------------------------
def build_constraints(
    problem: CadmmProblem,
    reg: BlockRegistry,
    z_blocks: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, List[object]]:
    """
    Assemble constraints for each block in the registry.

    Parameters
    ----------
    problem:
        Frozen CadmmProblem snapshot (no env/robot dependency).
    reg:
        BlockRegistry defining which macro blocks exist and their shapes.
    z_blocks:
        Optional dict of current block values (e.g., used for coupled constraints).
        For example, sigma's feasible set may depend on y_hat.

    Returns
    -------
    dict
        Mapping {block_name: [constraint1, constraint2, ...]}.
        Keys are exactly reg.names(). Each constraint is a dataclass instance from data.py.

    Notes
    -----
    * This stage implements simplified constraints for pipeline validation.
    * Complex coupled constraints are left as placeholders with stable interfaces.
    """
    z_blocks = {} if z_blocks is None else dict(z_blocks)
    out: Dict[str, List[object]] = {name: [] for name in reg.names()}
    flags = getattr(problem, "flags", None)

    # common parameters
    cap: Optional[np.ndarray] = None
    if ("f_hat" in reg.names()) or ("B_hat" in reg.names()):
        cap = _get_capacity_vector(problem, reg)
    
    sigma_max = float(_get_param(problem, "sigma_max", default=1.0))
    r_min = float(_get_param(problem, "r_min", default=0.0))
    r_max = float(_get_param(problem, "r_max", default=1.0))
    R_total = _get_param(problem, "R_total", default=None)
    B_total = _get_param(problem, "B_total", default=None)

    # per-block constraints
    for name in reg.names():
        dim = dim_of_block(reg, name)
        if name == "pos":
            # stage 1: no geometric constraints, pos is handled in q-step enumeration
            out[name] = []
        elif name == "cov":
            out[name] = [make_box(0.0, 1.0, dim)]
        
        elif name == "f_hat":
            # Flow per-link box: 0 <= f_hat <= capacity
            if cap is None:
                raise ValueError("Missing capacity: problem.link.capacity is required when f_hat block exists")
            hi = _ensure_len(cap, dim, label="capacity for f_hat")
            # Scheme-2 coupled feasibility: optionally enforce flow <= budget (edge-wise)
            # by tightening f_hat's upper bound using current B_hat.
            #
            # NOTE: This is a *z-step* feasibility coupling. It is intentionally gated by
            # a flag and defaults to OFF to preserve Stage D behavior.
            if bool(getattr(flags, "enable_flow_coupled_to_budget", False)):
                B_hat = z_blocks.get("B_hat", None)
                if B_hat is not None:
                    try:
                        Bv = np.asarray(B_hat, dtype=np.float32).reshape(-1)
                        Bv = _ensure_len(Bv, dim, label="B_hat for f_hat")
                        hi = np.minimum(hi, np.maximum(Bv, 0.0))
                    except Exception:
                        pass
            out[name] = [make_box(0.0, hi, dim)]
        elif name == "B_hat":
            # Budget box: 0 <= B_hat <= capacity
            if cap is None:
                raise ValueError("Missing capacity: problem.link.capacity is required when B_hat block exists")
            hi = _ensure_len(cap, dim, label="capacity for B_hat")
            cons: List[object] = [make_box(0.0, hi, dim)]

            budget = float(np.sum(hi)) if B_total is None else float(B_total)
            w = np.ones((dim,), dtype=np.float32)
            cons.append(make_halfspace(w=w, b=budget))
            out[name] = cons
        elif name == "y_hat":
            # Stage 1: simplex with sum=1
            cons = [make_simplex(dim=dim, sum_to=1.0)]
            cons.append(make_box(0.0, 1.0, dim))
            out[name] = cons
        elif name == "sigma":
            if bool(getattr(flags, "enable_sigma_coupled_to_y", True)):
                y_hat = z_blocks.get("y_hat", None)
                if y_hat is None:
                   out[name] = [make_box(0.0, sigma_max, dim)]
                else:
                    try:
                        lo = coverage_metrics.sigma_lower_bound_from_y_hat(
                            np.asarray(y_hat, dtype=np.float32).reshape(-1),
                            getattr(problem, "params", None),
                            flags
                        )
                        out[name] = [make_box(lo, sigma_max, dim)]
                    except Exception:
                        out[name] = [make_box(0.0, sigma_max, dim)]
        elif name == "r_hat":
            cons = [make_box(r_min, r_max, dim)]
            if R_total is not None:
                w = np.ones((dim,), dtype=np.float32)
                cons.append(make_halfspace(w=w, b=float(R_total)))
            out[name] = cons
        else:
            # unknown blocks: no constraints by default
            out[name] = []
    proj_gate = {
        "pos": "enable_proj_pos",
        "f_hat": "enable_proj_f_hat",
        "B_hat": "enable_proj_B_hat",
        "y_hat": "enable_proj_y_hat",
        "sigma": "enable_proj_sigma",
        "r_hat": "enable_proj_r_hat",
    }

    for blk, attr in proj_gate.items():
        if blk in out and hasattr(flags, attr) and (not getattr(flags, attr)):
            out[blk] = []


    validate_constraints(out, reg)
    return out

# ---------------------------------------------------------------------------
# Helpers 
# ---------------------------------------------------------------------------
def dim_of_block(reg: BlockRegistry, name: str) -> int:
    """Return flattened dimension of a block."""
    shp = reg.shape(name)
    return int(np.prod(shp, dtype=np.int64))

def make_box(lo: Any, hi: Any, dim: int) -> BoxC:
    """
    Create a Box constraint with broadcasting.

    lo, hi can be scalars or array-like, and will be broadcast/validated to (dim,).
    """
    lo_v = _broadcast_vec(lo, dim, "box.lo")
    hi_v = _broadcast_vec(hi, dim, "box.hi")
    if np.any(lo_v > hi_v):
        raise ValueError("Box constraint invalid: some lo > hi")
    return _construct_dataclass(BoxC, {"lo": lo_v, "hi": hi_v}, kind="BoxC")

def make_simplex(dim: int, sum_to: float=1.0) -> SimplexC:
    """Create a simplex constraint: x >= 0 and sum(x) = sum_to"""
    if dim < 0:
        raise ValueError(f"Simplex dim must be positive, got {dim}")
    if not np.isfinite(sum_to) or sum_to <= 0:
        raise ValueError(f"Simplex sum_to must be positive finite, got {sum_to}")
    return _construct_dataclass(SimplexC, {"dim": int(dim), "sum_to": float(sum_to)}, kind="SimplexC")

def make_halfspace(w: Any, b: float) -> HalfspaceC:
    """Create a halfspace constraint: w^T x <= b."""
    wv = np.asarray(w, dtype=np.float32).reshape(-1)
    if wv.ndim != 1 or wv.size == 0:
        raise ValueError("Halfspace w must be a non-empty 1-D vector")
    if not np.isfinite(b):
        raise ValueError(f"Halfspace b must be finite, got {b}")
    return _construct_dataclass(HalfspaceC, {"w": wv, "b": float(b)}, kind="HalfspaceC")

def make_affine(A: Any, b: Any) -> AffineC:
    """Create an affine equality constraint: A x = b."""
    Am = np.asarray(A, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32).reshape(-1)
    if Am.ndim != 2:
        raise ValueError("Affine A must be 2-D")
    if bv.ndim != 1:
        raise ValueError("Affine b must be 1-D")
    if Am.shape[0] != bv.size:
        raise ValueError(f"Affine shape mismatch: A is {Am.shape}, b is {bv.shape}")
    return _construct_dataclass(AffineC, {"A": Am, "b": bv}, kind="AffineC")

def validate_constraints(constraints_by_block: Mapping[str, List[object]], reg: BlockRegistry) -> None:
    """Validate assembled constraints: dimension consistency and basic shape checks.

    Raises ValueError if any constraint shape does not match its block dimension.
    """
    reg_names = set(reg.names())
    extra = set(constraints_by_block.keys()) - reg_names
    if extra:
        raise ValueError(f"constraints_by_block contains unknown block keys: {sorted(extra)}")

    for name, cons_list in constraints_by_block.items():
        dim = dim_of_block(reg, name)
        if cons_list is None:
            raise ValueError(f"constraints list for '{name}' must not be None")
        for c in cons_list:
            _validate_one_constraint(c, dim, block=name)

# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------
def _get_param(problem: Any, name: str, default: Any):
    """Fetch a parameter from problem with multiple common locations."""
    if hasattr(problem, name):
        return getattr(problem, name)
    params = getattr(problem, "params", None)
    if params is not None and hasattr(params, name):
        return getattr(params, name)
    flags = getattr(problem, "flags", None)
    if flags is not None and hasattr(flags, name):
        return getattr(flags, name)
    return default

def _get_capacity_vector(problem: Any, reg: BlockRegistry) -> np.ndarray:
    """
    Extract link capacity vector used for f_hat/B_hat bounds.

    Expected: problem.link.capacity is either scalar or array-like length E.
    """
    link = getattr(problem, "link", None)
    cap = getattr(link, "capacity", None)
    if cap is None:
        raise ValueError("CadmmProblem must provide link capacity via problem.link.capacity (scalar or vector).")
    cap_arr = np.asarray(cap, dtype=np.float32).reshape(-1)
    if cap_arr.size == 1:
        E = None
        if "f_hat" in reg.names():
            E = dim_of_block(reg, "f_hat")
        elif "B_hat" in reg.names():
            E = dim_of_block(reg, "B_hat")
        if E is not None:
            cap_arr = np.full((E,), float(cap_arr.item()), dtype=np.float32)
    
    if np.any(cap_arr < 0) or not np.all(np.isfinite(cap_arr)):
        raise ValueError("capacity must be finite and non-negative")
    return cap_arr

def _ensure_len(vec: np.ndarray, dim:int, *, label:str) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    if v.size == 1:
        return np.full((dim,), float(v.item()), dtype=np.float32)
    if v.size != dim:
        raise ValueError(f"{label} length mismatch: got {v.size}, expected {dim}")
    return v

def _broadcast_vec(x: Any, dim: int, label: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size == 1:
        return np.full((dim,), float(arr.item()), dtype=np.float32)
    if arr.size != dim:
        raise ValueError(f"{label} length mismatch: got {arr.size}, expected {dim}")
    return arr

def _construct_dataclass(cls: Any, kwargs: Dict[str, Any], *, kind: str) -> Any:
    """
    Construct a constraint dataclass.

    Preferred canonical field names:
      BoxC(lo, hi)
      SimplexC(dim, sum_to)
      HalfspaceC(w, b)
      AffineC(A, b)
      PolytopeC(G, h, lo, hi)
    """
    if cls is None:
        return RuntimeError(f"Constraint class for {kind} is None")
    # Fast path: try canonical kwargs directly.
    try:
        return cls(**kwargs)
    except TypeError:
        pass

    if not is_dataclass(cls):
        raise TypeError(
            f"{kind} must be a dataclass in data.py, or accept canonical kwargs {sorted(kwargs.keys())}."
        )
    
    fset = {f.name for f in fields(cls)}
    alias_groups = {
        "lo": ["lo", "lb", "lower", "l"],
        "hi": ["hi", "ub", "upper", "u"],
        "dim": ["dim", "n", "D"],
        "sum_to": ["sum_to", "sum", "s", "budget", "total"],
        "w": ["w", "a", "n", "normal"],
        "b": ["b", "rhs", "c"],
        "A": ["A", "mat", "M"],
        "G": ["G", "A", "mat"],
        "h": ["h", "b", "rhs"],
    }
    mapped: Dict[str, Any] = {}
    for k, v in kwargs.items():
        if k in fset:
            mapped[k] = v
            continue
        if k in alias_groups:
            for alt in alias_groups[k]:
                if alt in fset:
                    mapped[alt] = v
                    break
    try:
        return cls(**mapped)
    except TypeError as e:
        raise TypeError(
            f"Failed to construct {kind}. Tried canonical keys={sorted(kwargs.keys())} "
            f"and mapped keys={sorted(mapped.keys())}. "
            f"Your {kind} dataclass fields are {sorted(fset)}."
        ) from e
    
def _validate_one_constraint(c: object, dim: int, *, block: str) -> None:
    """Best-effort validation based on common constraint attributes."""
    cls_name = c.__class__.__name__

    # ---- Box ----
    lo, hi = _try_get_box_bounds(c)
    if lo is not None and hi is not None:
        if lo.size != dim or hi.size != dim:
            raise ValueError(
                f"BoxC dim mismatch for block '{block}': lo/hi len ({lo.size},{hi.size}) vs dim {dim}"
            )
        return

    # ---- Simplex ----
    sdim, ssum = _try_get_simplex_params(c)
    if sdim is not None:
        if int(sdim) != dim:
            raise ValueError(f"SimplexC dim mismatch for block '{block}': got {sdim} vs dim {dim}")
        if ssum is not None and (not np.isfinite(ssum) or float(ssum) <= 0):
            raise ValueError(f"SimplexC sum_to invalid for block '{block}': {ssum}")
        return

    # ---- Halfspace ----
    w, b = _try_get_halfspace_params(c)
    if w is not None and b is not None:
        if w.size != dim:
            raise ValueError(f"HalfspaceC dim mismatch for block '{block}': w len {w.size} vs dim {dim}")
        if not np.isfinite(b):
            raise ValueError(f"HalfspaceC b invalid for block '{block}': {b}")
        return

    # ---- Affine ----
    A, bb = _try_get_affine_params(c)
    if A is not None and bb is not None:
        if A.ndim != 2 or A.shape[1] != dim:
            raise ValueError(f"AffineC dim mismatch for block '{block}': A.shape {A.shape} vs dim {dim}")
        if bb.ndim != 1 or bb.size != A.shape[0]:
            raise ValueError(
                f"AffineC rhs mismatch for block '{block}': b len {bb.size} vs A rows {A.shape[0]}"
            )
        return

    # ---- Polytope ----
    Gm, hv = _try_get_polytope_params(c)
    if Gm is not None and hv is not None:
        if Gm.ndim != 2 or Gm.shape[1] != dim:
            raise ValueError(f"PolytopeC dim mismatch for block '{block}': G.shape {Gm.shape} vs dim {dim}")
        if hv.ndim != 1 or hv.size != Gm.shape[0]:
            raise ValueError(
                f"PolytopeC rhs mismatch for block '{block}': h len {hv.size} vs G rows {Gm.shape[0]}"
            )
        return

    raise ValueError(
        f"Unknown constraint object for block '{block}' (class={cls_name}). "
        "Add a validator branch in validate_constraints()."
    )

def _try_get_box_bounds(c: object):
    for lo_name, hi_name in [("lo", "hi"), ("lb", "ub"), ("lower", "upper"), ("l", "u")]:
        if hasattr(c, lo_name) and hasattr(c, hi_name):
            lo = np.asarray(getattr(c, lo_name), dtype=np.float32).reshape(-1)
            hi = np.asarray(getattr(c, hi_name), dtype=np.float32).reshape(-1)
            return lo, hi
    return None, None


def _try_get_simplex_params(c: object):
    dim = None
    sum_to = None
    for dn in ("dim", "n", "D"):
        if hasattr(c, dn):
            dim = int(getattr(c, dn))
            break
    for sn in ("sum_to", "sum", "s", "budget", "total"):
        if hasattr(c, sn):
            sum_to = float(getattr(c, sn))
            break
    return dim, sum_to


def _try_get_halfspace_params(c: object):
    w = None
    b = None
    for wn in ("w", "a", "n", "normal"):
        if hasattr(c, wn):
            w = np.asarray(getattr(c, wn), dtype=np.float32).reshape(-1)
            break
    for bn in ("b", "rhs", "c"):
        if hasattr(c, bn):
            b = float(getattr(c, bn))
            break
    return w, b


def _try_get_affine_params(c: object):
    A = None
    b = None
    for an in ("A", "mat", "M"):
        if hasattr(c, an):
            A = np.asarray(getattr(c, an), dtype=np.float32)
            break
    for bn in ("b", "rhs", "c"):
        if hasattr(c, bn):
            b = np.asarray(getattr(c, bn), dtype=np.float32).reshape(-1)
            break
    return A, b


def _try_get_polytope_params(c: object):
    Gm = None
    h = None
    for gn in ("G", "A", "mat"):
        if hasattr(c, gn):
            Gm = np.asarray(getattr(c, gn), dtype=np.float32)
            break
    for hn in ("h", "b", "rhs"):
        if hasattr(c, hn):
            h = np.asarray(getattr(c, hn), dtype=np.float32).reshape(-1)
            break
    return Gm, h

