"""
scripts.core.inner.projections

Projection computation layer for TRACE / Inner C-ADMM.

Scope (STRICT)
--------------
1) This module performs ONLY projection computations onto convex constraint sets,
   operating on *flat vectors* v: (d,) -> x: (d,).
2) It does NOT depend on env_state / robot / map / cost.
3) It consumes constraint description objects produced by constraints.py:
   constraints_by_block[name] = [C1, C2, ...] (list denotes intersection)

Supported constraint dataclasses (from data.py)
----------------------------------------------
- BoxC
- SimplexC
- WeightedSimplexC
- HalfspaceC
- AffineC
- PolytopeC

Intersection projection methods
------------------------------
- serial  : cyclic sequential projections (P_Ck(...P_C2(P_C1(v))...))
- dykstra : Dykstra's algorithm (recommended for intersections)

Diagnostics (info dict)
-----------------------
project_to_constraints returns (x, info) where info includes:
- iters_used
- violations: list[float] (per constraint, same order)
- max_violation
- delta_norm (optional)

Block-level API
---------------
project_blocks(z_tilde_blocks, constraints_by_block, ...) projects each block
independently by flattening -> projecting -> reshaping back.
"""

from __future__ import annotations

from dataclasses import is_dataclass, fields
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


from scripts.core.data import (
    BoxC,
    SimplexC,
    WeightedSimplexC,
    HalfspaceC,
    AffineC,
    PolytopeC
)


# ---------------------------------------------------------------------------
# Atomic projections + violations
# ---------------------------------------------------------------------------

def proj_box(v: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    v = _as1d(v)
    lo = _as1d(lo)
    hi = _as1d(hi)
    if lo.size != v.size or hi.size != v.size:
        raise ValueError(f"proj_box dim mismatch: v={v.size}, lo={lo.size}, hi={hi.size}")
    return np.minimum(np.maximum(v, lo), hi)

def viol_box(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    x = _as1d(x)
    lo = _as1d(lo)
    hi = _as1d(hi)
    if lo.size != x.size or hi.size != x.size:
        raise ValueError(f"proj_box dim mismatch: x={x.size}, lo={lo.size}, hi={hi.size}")
    v1 = np.maximum(lo - x, 0.0)
    v2 = np.minimum(x - hi, 0.0)
    return float(np.max(np.maximum(v1, v2)))

def proj_simplex(v: np.ndarray, s: float) -> np.ndarray:
    """Project onto simplex {x >= 0, sum(x) = s}, s > 0. O(nlogn) sorting algorithm"""
    v = _as1d(v).astype(np.float64, copy=False)
    n = v.size
    if n == 0:
        return v.astype(np.float32)
    if not np.isfinite(s) or s <= 0:
        raise ValueError(f"Simplex sum s must be positive finite, got {s}")
    
    # Sort descending
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - s
    # Find rho = max { j | u_j - (cssv_j - s) / j > 0 }
    j = np.arange(1, n + 1, dtype=np.float64)
    cond = u - (cssv - s) / j > 0
    if not np.any(cond):
        theta = (cssv[1] - s) / n
    else:
        rho = int(np.max(np.where(cond)[0]))
        theta = (cssv[rho] - s) / float(rho + 1)
    x = np.maximum(v - theta, 0.0)
    # enforce exact sum within numeric tolerance by rescaling tiny drift
    if x.sum() > 0:
        x *= (s / x.sum())
    return x.astype(np.float32)

def viol_simplex(x: np.ndarray, s: float) -> float:
    x = _as1d(x)
    if not np.isfinite(s) or s <= 0:
        raise ValueError(f"Simplex sum s must be positive finite, got {s}")
    neg = float(np.maximum(-np.min(x), 0.0))
    eq = float(abs(np.sum(x) - s))
    return max(neg, eq)

def proj_weighted_simplex(v: np.ndarray, w: np.ndarray, s: float, *, 
                          tol: float = 1e-10, max_iters: int = 200) -> np.ndarray:
    """
    Project onto weighted simplex {x>=0, w^T x = s}, w>0 componentwise.

    Uses bisection on lambda for solution x = max(0, v - lambda*w).
    Allows lambda to be negative (x can exceed v).
    """
    v = _as1d(v).astype(np.float64, copy=False)
    w = _as1d(w).astype(np.float64, copy=False)
    if v.size != w.size:
        raise ValueError(f"WeightedSimplex dim mismatch: v={v.size}, w={w.size}")
    if not np.isfinite(s) or s <= 0:
        raise ValueError(f"WeightedSimplex sum s must be positive finite, got {s}")
    if np.any(~np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("WeightedSimplex requires w to be finite and strictly positive")

    def f(lam: float) -> float:
        x = np.maximum(0.0, v - lam * w)
        return float(np.dot(w, x) - s)
    
    # Find bracket [lo, hi] such that f(lo) >= 0 and f(hi) <= 0
    lo = 0.0
    hi = 0.0
    f0 = f(0.0)
    if abs(f0) <= tol:
        x0 = np.maximum(0.0, v)
        # adjust small drift
        scale = s / float(np.dot(w, x0)) if np.dot(w, x0) > 0 else 1.0
        return (x0 * scale).astype(np.float32)
    
    if f0 > 0:
        # need larger lambda to reduce dot(w,x)
        lo = 0.0
        hi = 1.0
        while f(hi) > 0 and hi < 1e-12:
            hi *= 2.0
    else:
        # f0 < 0, need negative lambda to increase dot(w,x)
        hi = 0.0
        lo = -1.0
        while f(lo) < 0 and abs(lo) < 1e-12:
            lo *= 2.0
    flo = f(lo)
    fhi = f(hi)
    if flo < 0.0 or fhi > 0.0:
        raise RuntimeError("Failed to bracket lambda for weighted simplex projection")
    
    for _ in range(max_iters):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) <= tol:
            lo = hi = mid
            break
        if fm > 0.0:
            lo = mid
        else:
            hi = mid
    lam = 0.5 * (lo + hi)
    x = np.maximum(0.0, v - lam * w)
    wx = float(np.dot(w, x))
    if wx > 0:
        x *= (s / wx)
    return x.astype(np.float32)

def viol_weighted_simplex(x: np.ndarray, w: np.ndarray, s: float) -> float:
    x = _as1d(x)
    w = _as1d(w)
    if x.size != w.size:
        raise ValueError(f"viol_weighted_simplex dim mismatch: x={x.size}, w={w.size}")
    neg = float(np.maximum(-np.min(x), 0.0))
    eq = float(abs(np.dot(w, x) - s))
    return max(neg, eq)

def proj_halfspace(v: np.ndarray, a: np.ndarray, b: float) -> np.ndarray:
    """Project onto halfspace {a^T x <= b} with closed form."""
    v = _as1d(v)
    a = _as1d(a)
    if a.size != v.size:
        raise ValueError(f"proj_halfspace dim mismatch: v={v.size}, a={a.size}")
    if not np.isfinite(b):
        raise ValueError(f"Halfspace b must be finite, got {b}")
    av = float(np.dot(a, v))
    if av <= b:
        return v.copy()
    aa = float(np.dot(a, a))
    if aa <= 0:
        raise ValueError("Halfspace normal a must be nonzero")
    t = (av - b) / aa
    return (v - t * a).astype(np.float32, copy=False)

def viol_halfspace(x: np.ndarray, a: np.ndarray, b: float) -> float:
    x = _as1d(x)
    a = _as1d(a)
    if a.size != x.size:
        raise ValueError(f"viol_halfspace dim mismatch: x={x.size}, a={a.size}")
    return float(max(np.dot(a, x) - b, 0.0))

def proj_affine(v: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Project onto affine set {A x = b}.

    Uses minimum-norm correction:
        Solve (A A^T) lambda = A v - b  (via lstsq/pinv)
        x = v - A^T lambda
    """
    v = _as1d(v).astype(np.float64, copy=False)
    A = np.asarray(A, dtype=np.float64)
    b = _as1d(b).astype(np.float64, copy=False)
    if A.ndim != 2:
        raise ValueError("Affine A must be 2-D")
    if A.shape[1] != v.size:
        raise ValueError(f"Affine dim mismatch: A.shape={A.shape}, v.size={v.size}")
    if A.shape[0] != b.size:
        raise ValueError(f"Affine rhs mismatch: A rows={A.shape[0]}, b.size={b.size}")
    
    r =  A @ v - b
    M = A @ A.T
    # robust sovle for possibly rank-deficient M
    lam, *_ = np.linalg.lstsq(M, r, rcond=None)
    x = v - A.T @ lam
    return x.astype(np.float32)
    
def viol_affine(x: np.ndarray, A: np.ndarray, b: np.ndarray) -> float:
    x = _as1d(x).astype(np.float64, copy=False)
    A = np.asarray(A, dtype=np.float64)
    b = _as1d(b).astype(np.float64, copy=False)
    if A.ndim != 2 or A.shape[1] != x.size or A.shape[0] != b.size:
        raise ValueError(f"viol_affine dim mismatch: A={A.shape}, x={x.size}, b={b.size}")
    r = A @ x - b
    return float(np.linalg.norm(r, ord=2))

def proj_polytope(
    v: np.ndarray, 
    G: np.ndarray,
    h: np.ndarray,
    lo: Optional[np.ndarray] = None,
    hi: Optional[np.ndarray] = None,
    *,
    method: str = "dykstra",
    iters: int = 200,
    tol: float = 1e-6,
    pgd_step: float = 0.2,
    pgd_penalty: float = 10.0
) -> np.ndarray:
    """
    Project onto polytope {Gx <= h} optionally with box lo<=x<=hi.

    No external QP solver is required.
    - method="dykstra"/"serial": use intersection projections over halfspaces (+ box)
    - method="pgd": hinge-loss PGD with optional box projection each step
    """
    v = _as1d(v).astype(np.float32, copy=False)
    G = np.asarray(G, dtype=np.float32)
    h = _as1d(h).astype(np.float32, copy=False)
    if G.ndim != 2:
        raise ValueError("Polytope G must be 2-D")
    if G.shape[1] != v.size:
        raise ValueError(f"Polytope dim mismatch: G.shape={G.shape}, v.size={v.size}")
    if G.shape[0] != h.size:
        raise ValueError(f"Polytope rhs mismatch: G rows={G.shape[0]}, h.size={h.size}")

    lo_v = _as1d(lo).astype(np.float32, copy=False) if lo is not None else None
    hi_v = _as1d(hi).astype(np.float32, copy=False) if hi is not None else None
    if lo_v is not None and lo_v.size != v.size:
        raise ValueError("Polytope lo dim mismatch")
    if hi_v is not None and hi_v.size != v.size:
        raise ValueError("Polytope hi dim mismatch")
    
    m = G.shape[0]
    method = method.lower()

    if method in ("dykstra", "serial"):
        cons: List[object] = []
        if lo_v is not None or hi_v is not None:
            if lo_v is None:
                lo_v = np.full((v.size,), -np.inf, dtype=np.float32)
            if hi_v is None:
                hi_v = np.full((v.size,), np.inf, dtype=np.float32)
            # create a BoxC-like object via direct projection
            cons.append(("__box__", lo_v, hi_v))
        for i in range(m):
            cons.append(("__halfspace__", G[i], float(h[i])))
        # use internal Dykstra with these lightweight tuples
        x, _ = project_to_constraints(v, cons, method=method, iters=iters, tol=tol)
        return x
    if method == "pgd":
        x = v.copy()
        if lo_v is not None or hi_v is not None:
            if lo_v is None:
                lo_v = np.full((v.size,), -np.inf, dtype=np.float32)
            if hi_v is None:
                hi_v = np.full((v.size,), np.inf, dtype=np.float32)
            x = proj_box(x, lo_v, hi_v)
        for _ in range(iters):
            # grad of 0.5 || x - v || ^ 2 is (x - v)
            grad = (x - v).astype(np.float32, copy=False)
            # hinge loss: 0.5 * sum(max(Gx - h, 0) ^ 2)
            s = (G @ x - h)
            p = np.maximum(s, 0.0)
            if float(np.max(p)) <= tol:
                break
            grad = grad + pgd_penalty * (G.T @ p).astype(np.float32)
            x_new = x - pgd_step * grad
            if lo_v is not None or hi_v is not None:
                x_new = proj_box(x_new, lo_v, hi_v)
            if np.linalg.norm(x_new - x) <= tol:
                x = x_new
                break
            x = x_new
        return x
    
    raise ValueError(f"Unknown polytope projection method: {method}")

def viol_polytope(
    x: np.ndarray,
    G: np.ndarray,
    h: np.ndarray,
    lo: Optional[np.ndarray] = None,
    hi: Optional[np.ndarray] = None,
) -> float:
    x = _as1d(x).astype(np.float32, copy=False)
    G = np.asarray(G, dtype=np.float32)
    h = _as1d(h).astype(np.float32, copy=False)
    if G.ndim != 2 or G.shape[1] != x.size or G.shape[0] != h.size:
        raise ValueError("viol_polytope dim mismatch")
    ineq = np.maximum(G @ x - h, 0.0)
    maxv = float(np.max(ineq)) if ineq.size > 0 else 0.0
    if lo is not None or hi is not None:
        lo_v = _as1d(lo).astype(np.float32) if lo is not None else np.full((x.size,), -np.inf, np.float32)
        hi_v = _as1d(hi).astype(np.float32) if hi is not None else np.full((x.size,), np.inf, np.float32)
        maxv = max(maxv, viol_box(x, lo_v, hi_v))
    return float(maxv)

# ---------------------------------------------------------------------------
# Constraint object dispatch (from dataclasses) + lightweight tuple support
# ---------------------------------------------------------------------------

def project_constraint(v: np.ndarray, c: object, *, poly_method: str = "dykstra"):
    """Project v onto a single constraint object c."""
    v = _as1d(v).astype(np.float32, copy=False)

    # lightweight internal tuples for polytope decomposition
    if isinstance(c, tuple) and len(c) > 0:
        tag = c[0]
        if tag == "__box__":
            lo, hi = c[1], c[2]
            return proj_box(v, lo, hi)
        if tag == "__halfspace__":
            a, b = c[1], c[2]
            return proj_halfspace(v, a, float(b))
        raise ValueError(f"Unknown internal constraint tuple tag: {tag}")
    
    # dataclass-based constraints
    if isinstance(c, BoxC) or c.__class__.__name__ == "BoxC":
        lo, hi = _get_box_fields(c)
        return proj_box(v, lo, hi)

    if isinstance(c, SimplexC) or c.__class__.__name__ == "SimplexC":
        s = _get_simplex_sum(c)
        return proj_simplex(v, s)

    if isinstance(c, WeightedSimplexC) or c.__class__.__name__ == "WeightedSimplexC":
        w, s = _get_weighted_simplex_fields(c)
        return proj_weighted_simplex(v, w, s)

    if isinstance(c, HalfspaceC) or c.__class__.__name__ == "HalfspaceC":
        a, b = _get_halfspace_fields(c)
        return proj_halfspace(v, a, b)

    if isinstance(c, AffineC) or c.__class__.__name__ == "AffineC":
        A, b = _get_affine_fields(c)
        return proj_affine(v, A, b)

    if isinstance(c, PolytopeC) or c.__class__.__name__ == "PolytopeC":
        G, h, lo, hi = _get_polytope_fields(c)
        return proj_polytope(v, G, h, lo=lo, hi=hi, method=poly_method)

    raise TypeError(f"Unsupported constraint object type: {type(c)}")


def constraint_violation(x: np.ndarray, c: object) -> float:
    """Compute scalar violation of x w.r.t. constraint c (0 means satisfied)."""
    x = _as1d(x).astype(np.float32, copy=False)

    if isinstance(c, tuple) and len(c) >= 1:
        tag = c[0]
        if tag == "__box__":
            lo, hi = c[1], c[2]
            return viol_box(x, lo, hi)
        if tag == "__halfspace__":
            a, b = c[1], c[2]
            return viol_halfspace(x, a, float(b))
        raise ValueError(f"Unknown internal constraint tuple tag: {tag}")

    if isinstance(c, BoxC) or c.__class__.__name__ == "BoxC":
        lo, hi = _get_box_fields(c)
        return viol_box(x, lo, hi)

    if isinstance(c, SimplexC) or c.__class__.__name__ == "SimplexC":
        s = _get_simplex_sum(c)
        return viol_simplex(x, s)

    if isinstance(c, WeightedSimplexC) or c.__class__.__name__ == "WeightedSimplexC":
        w, s = _get_weighted_simplex_fields(c)
        return viol_weighted_simplex(x, w, s)

    if isinstance(c, HalfspaceC) or c.__class__.__name__ == "HalfspaceC":
        a, b = _get_halfspace_fields(c)
        return viol_halfspace(x, a, b)

    if isinstance(c, AffineC) or c.__class__.__name__ == "AffineC":
        A, b = _get_affine_fields(c)
        return viol_affine(x, A, b)

    if isinstance(c, PolytopeC) or c.__class__.__name__ == "PolytopeC":
        G, h, lo, hi = _get_polytope_fields(c)
        return viol_polytope(x, G, h, lo=lo, hi=hi)

    raise TypeError(f"Unsupported constraint object type: {type(c)}")

# ---------------------------------------------------------------------------
# Intersection projection
# ---------------------------------------------------------------------------

def project_to_constraints(
    v: np.ndarray,
    constraints: Sequence[object],
    *,
    method: str = "dykstra",
    iters: int = 50,
    tol: float = 1e-6,
    poly_method: str = "dykstra",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Project v onto intersection of constraints list.

    Parameters
    ----------
    v : (d,)
    constraints : list of constraint objects; list denotes intersection
    method : "serial" or "dykstra"
    iters : max outer iterations (cycles)
    tol : stopping tolerance (max_violation and change)
    poly_method : method for PolytopeC inner projection ("dykstra"/"serial"/"pgd")

    Returns
    -------
    x, info
    """
    v = _as1d(v).astype(np.float32, copy=False)
    cons = list(constraints)

    if len(cons) == 0:
        return v.copy(), {"iters_used": 0, "violations": [], "max_violation": 0.0, "delta_norm": 0.0}

    method = method.lower()
    if method not in ("serial", "dykstra"):
        raise ValueError(f"Unknown intersection projection method: {method}")
    
    x = v.copy()
    prev = x.copy()

    if method == "serial":
        used = 0
        for k in range(iters):
            for c in cons:
                x = project_constraint(x, c, poly_method=poly_method)
            used = k + 1
            viols = [constraint_violation(x, c) for c in cons]
            maxv = float(np.max(viols)) if viols else 0.0
            delta = float(np.linalg.norm(x - prev))
            if maxv <= tol and delta <= tol:
                break
            prev = x.copy()
        info = {"iters_used": used, "violations": viols, "max_violation": maxv, "delta_norm": delta}
        return x, info

    # Dykstra
    p = [np.zeros_like(x) for _ in cons]
    used = 0
    for k in range(iters):
        x_old = x.copy()
        for i, c in enumerate(cons):
            y = x + p[i]
            x_new = project_constraint(y, c, poly_method=poly_method)
            p[i] = y - x_new
            x = x_new
        used = k + 1
        viols = [constraint_violation(x, c) for c in cons]
        maxv = float(np.max(viols)) if viols else 0.0
        delta = float(np.linalg.norm(x - x_old))
        if maxv <= tol and delta <= tol:
            break
    info = {"iters_used": used, "violations": viols, "max_violation": maxv, "delta_norm": delta}
    return x, info

# ---------------------------------------------------------------------------
# Block-level projection (z-step entry)
# ---------------------------------------------------------------------------

def project_blocks(
    z_tilde_blocks: Dict[str, np.ndarray],
    constraints_by_block: Dict[str, List[object]],
    *,
    method: str = "dykstra",
    iters: int = 50,
    tol: float = 1e-6,
    poly_method: str = "dykstra",
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
    """Project each block independently using its constraint list (intersection).

    Rules:
    - Flatten -> project_to_constraints -> reshape back
    - If constraint list empty or missing -> return original
    """
    out: Dict[str, np.ndarray] = {}
    diag: Dict[str, Dict[str, Any]] = {}

    for name, arr in z_tilde_blocks.items():
        cons = constraints_by_block.get(name, [])
        arr_np = np.asarray(arr)
        shp = arr_np.shape
        v = arr_np.reshape(-1).astype(np.float32, copy=False)
        if cons is None or len(cons) == 0:
            out[name] = arr_np.copy()
            diag[name] = {"iters_used": 0, "violations": [], "max_violation": 0.0, "delta_norm": 0.0}
            continue
        x, info = project_to_constraints(v, cons, method=method, iters=iters, tol=tol, poly_method=poly_method)
        out[name] = x.reshape(shp).astype(arr_np.dtype, copy=False)
        diag[name] = info
    return out, diag

def project_one_block(
    arr_1d_or_nd: np.ndarray,
    cons_list: List[object],
    *,
    method: str = "dykstra",
    iters: int = 50,
    tol: float = 1e-6,
    poly_method: str = "dykstra"
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Project a single block array to an intersection of constraints.

    Helper to avoid duplicating flatten/project/reshape logic when some blocks
    require a second projection pass (e.g., sigma coupled to projected y_hat).
    """
    a = np.asarray(arr_1d_or_nd)
    shp = a.shape
    v = a.reshape(-1).astype(np.float32, copy=False)
    if cons_list is None or len(cons_list) == 0:
        return a.copy(), {"iters_used": 0, "violations": [], "max_violation": 0.0, "delta_norm": 0.0}
    x, info = project_to_constraints(v, cons_list, method=method, iters=iters, tol=tol, poly_method=poly_method)
    return x.reshape(shp).astype(a.dtype, copy=False), info

# ---------------------------------------------------------------------------
# Field extractors 
# ---------------------------------------------------------------------------

def _get_box_fields(c: object) -> Tuple[np.ndarray, np.ndarray]:
    for lo_name, hi_name in [("lo", "hi"), ("lb", "ub"), ("lower", "upper"), ("l", "u")]:
        if hasattr(c, lo_name) and hasattr(c, hi_name):
            lo = _as1d(getattr(c, lo_name)).astype(np.float32, copy=False)
            hi = _as1d(getattr(c, hi_name)).astype(np.float32, copy=False)
            return lo, hi
    raise AttributeError("BoxC must have (lo,hi) or equivalent fields")


def _get_simplex_sum(c: object) -> float:
    for sname in ("sum_to", "sum", "s", "budget", "total"):
        if hasattr(c, sname):
            s = float(getattr(c, sname))
            return s
    # sometimes simplex stores dim only; default to 1.0 is unsafe
    raise AttributeError("SimplexC must have sum_to (or alias) field")


def _get_weighted_simplex_fields(c: object) -> Tuple[np.ndarray, float]:
    w = None
    for wname in ("w", "weights", "weight"):
        if hasattr(c, wname):
            w = _as1d(getattr(c, wname)).astype(np.float32, copy=False)
            break
    if w is None:
        raise AttributeError("WeightedSimplexC must have w field (or alias)")

    s = None
    for sname in ("sum_to", "sum", "s", "budget", "total"):
        if hasattr(c, sname):
            s = float(getattr(c, sname))
            break
    if s is None:
        raise AttributeError("WeightedSimplexC must have sum_to (or alias) field")
    return w, float(s)


def _get_halfspace_fields(c: object) -> Tuple[np.ndarray, float]:
    a = None
    for aname in ("a", "w", "normal", "n"):
        if hasattr(c, aname):
            a = _as1d(getattr(c, aname)).astype(np.float32, copy=False)
            break
    if a is None:
        raise AttributeError("HalfspaceC must have a/w/normal field")

    b = None
    for bname in ("b", "rhs", "c"):
        if hasattr(c, bname):
            b = float(getattr(c, bname))
            break
    if b is None:
        raise AttributeError("HalfspaceC must have b/rhs field")
    return a, float(b)


def _get_affine_fields(c: object) -> Tuple[np.ndarray, np.ndarray]:
    A = None
    for aname in ("A", "mat", "M"):
        if hasattr(c, aname):
            A = np.asarray(getattr(c, aname), dtype=np.float32)
            break
    if A is None:
        raise AttributeError("AffineC must have A field (or alias)")
    b = None
    for bname in ("b", "rhs", "c"):
        if hasattr(c, bname):
            b = _as1d(getattr(c, bname)).astype(np.float32, copy=False)
            break
    if b is None:
        raise AttributeError("AffineC must have b field (or alias)")
    return A, b


def _get_polytope_fields(c: object) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    G = None
    for gname in ("G", "A", "mat"):
        if hasattr(c, gname):
            G = np.asarray(getattr(c, gname), dtype=np.float32)
            break
    if G is None:
        raise AttributeError("PolytopeC must have G field")
    h = None
    for hname in ("h", "b", "rhs"):
        if hasattr(c, hname):
            h = _as1d(getattr(c, hname)).astype(np.float32, copy=False)
            break
    if h is None:
        raise AttributeError("PolytopeC must have h field")

    lo = None
    hi = None
    for lname in ("lo", "lb", "lower"):
        if hasattr(c, lname):
            lo = _as1d(getattr(c, lname)).astype(np.float32, copy=False)
            break
    for hname in ("hi", "ub", "upper"):
        if hasattr(c, hname):
            hi = _as1d(getattr(c, hname)).astype(np.float32, copy=False)
            break
    return G, h, lo, hi


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _as1d(x: Any) -> np.ndarray:
    if x is None:
        return None  # type: ignore
    arr = np.asarray(x)
    return arr.reshape(-1)
