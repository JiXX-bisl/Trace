"""scripts.core.blocks

TRACE / Inner C-ADMM engineering layer: **variable protocol** and **residual protocol**.

This module is the *only* source of :class:`BlockRegistry`.

Unified I/O protocol (STRICT)
----------------------------

Let ``D = reg.total_dim``.

* Global consensus variable: ``z`` has shape ``(D,)``.
* Per-robot local variables: ``q`` and ``u`` have shape ``(N, D)``.

All block access **must** go through :class:`BlockRegistry` slices (no hard-coded
indices in solvers).

Block shapes are defined at the *macro block* level (e.g., ``pos`` is the stacked
positions of all robots, shape ``(2*N,)`` with a fixed flatten order).

Residual Model
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import abc

import numpy as np


# ---------------------------------------------------------------------------
# Import shared dataclasses from data.py (required by project structure)
# ---------------------------------------------------------------------------
from scripts.core.data import BlockDef, FeatureFlags  



# ---------------------------------------------------------------------------
# Block Registry
# ---------------------------------------------------------------------------


class BlockRegistry:
    """Manage macro blocks as slices over a flat vector.

    Parameters
    ----------
    blocks:
        Ordered list of :class:`BlockDef`. Names must be unique.

    Notes
    -----
    * ``total_dim`` is the sum of flattened dims of each block.
    * ``sl(name)`` returns the slice in the flat vector.
    * ``shape(name)`` returns the original shape.
    """

    def __init__(self, blocks: List[BlockDef], *, dtype: np.dtype = np.float32):
        if not isinstance(blocks, list) or len(blocks) == 0:
            raise ValueError("`blocks` must be a non-empty list[BlockDef].")

        self._dtype = np.dtype(dtype)
        self._blocks: List[BlockDef] = list(blocks)
        self._offsets: Dict[str, slice] = {}
        self._shapes: Dict[str, Tuple[int, ...]] = {}
        self._sizes: Dict[str, int] = {}

        seen = set()
        cursor = 0
        for b in self._blocks:
            if not hasattr(b, "name") or not hasattr(b, "shape"):
                raise TypeError("Each BlockDef must have fields: name (str) and shape (tuple[int,...]).")

            name = str(b.name)
            if name in seen:
                raise ValueError(f"Duplicate block name: '{name}'")
            seen.add(name)

            shape = _validate_shape_tuple(b.shape, name=name)
            size = int(np.prod(shape, dtype=np.int64))
            if size <= 0:
                raise ValueError(f"Invalid block '{name}' shape={shape}: flattened size must be > 0")

            sl = slice(cursor, cursor + size)
            self._offsets[name] = sl
            self._shapes[name] = shape
            self._sizes[name] = size
            cursor += size

        self.total_dim: int = int(cursor)

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def offsets(self) -> Dict[str, slice]:
        """Mapping name -> slice in the flat vector (returned as a shallow copy)."""
        return dict(self._offsets)

    def names(self) -> List[str]:
        return [b.name for b in self._blocks]

    def sl(self, name: str) -> slice:
        try:
            return self._offsets[name]
        except KeyError as e:
            raise KeyError(f"Unknown block name '{name}'. Available: {self.names()}") from e

    def shape(self, name: str) -> Tuple[int, ...]:
        try:
            return self._shapes[name]
        except KeyError as e:
            raise KeyError(f"Unknown block name '{name}'. Available: {self.names()}") from e

    def size(self, name: str) -> int:
        try:
            return self._sizes[name]
        except KeyError as e:
            raise KeyError(f"Unknown block name '{name}'. Available: {self.names()}") from e

    def pack(self, blocks_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Pack a dict of named blocks into a flat vector ``(D,)``.

        Strict checks:
        * all registry blocks must exist in ``blocks_dict``
        * each block must match its expected shape
        * extra keys in ``blocks_dict`` are rejected
        """

        if not isinstance(blocks_dict, dict):
            raise TypeError("pack expects a dict[str, np.ndarray]")

        extra = set(blocks_dict.keys()) - set(self._offsets.keys())
        if extra:
            raise KeyError(f"pack got unknown block keys: {sorted(extra)}")

        x = np.zeros((self.total_dim,), dtype=self._dtype)
        for name in self.names():
            if name not in blocks_dict:
                raise KeyError(f"pack missing required block '{name}'")
            arr = np.asarray(blocks_dict[name], dtype=self._dtype)
            exp_shape = self.shape(name)
            if tuple(arr.shape) != exp_shape:
                raise ValueError(f"Block '{name}' shape mismatch: got {arr.shape}, expected {exp_shape}")
            flat = np.ascontiguousarray(arr).reshape(-1)
            sl = self.sl(name)
            if flat.size != (sl.stop - sl.start):
                raise RuntimeError("Internal size mismatch (this should never happen)")
            x[sl] = flat
        return x

    def unpack(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """Unpack a flat vector ``(D,)`` into a dict of named blocks.

        Returned arrays are *views* into the (possibly casted) flat buffer.
        """

        x_arr = np.asarray(x)
        if x_arr.ndim != 1:
            raise ValueError(f"unpack expects a 1-D array of shape (D,), got shape={x_arr.shape}")
        if x_arr.size != self.total_dim:
            raise ValueError(f"unpack expects size {self.total_dim}, got {x_arr.size}")

        if x_arr.dtype != self._dtype:
            x_arr = x_arr.astype(self._dtype, copy=True)

        out: Dict[str, np.ndarray] = {}
        for name in self.names():
            sl = self.sl(name)
            out[name] = x_arr[sl].reshape(self.shape(name))
        return out


def _validate_shape_tuple(shape: object, *, name: str) -> Tuple[int, ...]:
    if not isinstance(shape, tuple):
        raise TypeError(f"Block '{name}' shape must be a tuple[int,...], got {type(shape)}")
    if len(shape) == 0:
        raise ValueError(f"Block '{name}' shape must be non-empty")
    out: List[int] = []
    for i, d in enumerate(shape):
        if not isinstance(d, (int, np.integer)):
            raise TypeError(f"Block '{name}' shape[{i}] must be int, got {type(d)}")
        d_int = int(d)
        if d_int <= 0:
            raise ValueError(f"Block '{name}' shape[{i}] must be > 0, got {d_int}")
        out.append(d_int)
    return tuple(out)


# ---------------------------------------------------------------------------
# Registry factory
# ---------------------------------------------------------------------------


def make_registry(N: int, E: int, G: int, flags: FeatureFlags) -> BlockRegistry:
    """Factory to build registry from problem sizes and FeatureFlags.

    Macro block shapes (default, can be disabled by flags.enable_*):

    * pos   : (2*N,)
    * cov   : (1,)
    * f_hat : (E,)
    * B_hat : (E,)
    * y_hat : (G,)
    * sigma : (G,)
    * r_hat : (N,)

    Turning off any macro block MUST reduce ``total_dim`` .
    """

    if N <= 0:
        raise ValueError(f"N must be positive, got {N}")
    if E < 0:
        raise ValueError(f"E must be non-negative, got {E}")
    if G < 0:
        raise ValueError(f"G must be non-negative, got {G}")

    def enabled(attr: str) -> bool:
        v = getattr(flags, attr)
        if not isinstance(v, bool):
            raise TypeError(f"flags.{attr} must be bool, got {type(v)}")
        return v

    blocks: List[BlockDef] = []
    # NOTE: keep deterministic order.
    if enabled("enable_pos"):
        blocks.append(BlockDef(name="pos", shape=(2 * N,)))
    if enabled("enable_cov"):
        blocks.append(BlockDef(name="cov", shape=(1,)))
    if enabled("enable_f_hat"):
        if E <= 0:
            raise ValueError("enable_f_hat=True but E<=0; would create an empty block")
        blocks.append(BlockDef(name="f_hat", shape=(E,)))
    if enabled("enable_B_hat"):
        if E <= 0:
            raise ValueError("enable_B_hat=True but E<=0; would create an empty block")
        blocks.append(BlockDef(name="B_hat", shape=(E,)))
    if enabled("enable_y_hat"):
        if G <= 0:
            raise ValueError("enable_y_hat=True but G<=0; would create an empty block")
        blocks.append(BlockDef(name="y_hat", shape=(G,)))
    if enabled("enable_sigma"):
        if G <= 0:
            raise ValueError("enable_sigma=True but G<=0; would create an empty block")
        blocks.append(BlockDef(name="sigma", shape=(G,)))
    if enabled("enable_r_hat"):
        blocks.append(BlockDef(name="r_hat", shape=(N,)))

    if not blocks:
        raise ValueError("All blocks are disabled by flags; registry would be empty.")

    return BlockRegistry(blocks)


# ---------------------------------------------------------------------------
# Block access helpers (x can be (D,) or (N,D))
# ---------------------------------------------------------------------------


def get_block(reg: BlockRegistry, x: np.ndarray, name: str) -> np.ndarray:
    """Return a reshaped view/array of the given block.

    Supports:
    * x shape (D,)  -> returns shape reg.shape(name)
    * x shape (N,D) -> returns shape (N,) + reg.shape(name)
    """

    sl = reg.sl(name)
    shp = reg.shape(name)
    x_arr = np.asarray(x)
    if x_arr.ndim == 1:
        if x_arr.size != reg.total_dim:
            raise ValueError(f"x has size {x_arr.size}, expected {reg.total_dim}")
        return x_arr[sl].reshape(shp)
    if x_arr.ndim == 2:
        if x_arr.shape[1] != reg.total_dim:
            raise ValueError(f"x has shape {x_arr.shape}, expected (N,{reg.total_dim})")
        N = x_arr.shape[0]
        return x_arr[:, sl].reshape((N,) + shp)
    raise ValueError(f"x must be 1-D or 2-D, got shape={x_arr.shape}")


def set_block(reg: BlockRegistry, x: np.ndarray, name: str, value: np.ndarray) -> None:
    """In-place set a block in x (strict shape checks)."""

    sl = reg.sl(name)
    shp = reg.shape(name)
    x_arr = np.asarray(x)
    val = np.asarray(value, dtype=x_arr.dtype)

    if x_arr.ndim == 1:
        if x_arr.size != reg.total_dim:
            raise ValueError(f"x has size {x_arr.size}, expected {reg.total_dim}")
        if tuple(val.shape) != shp:
            raise ValueError(f"value shape mismatch for block '{name}': got {val.shape}, expected {shp}")
        x_arr[sl] = np.ascontiguousarray(val).reshape(-1)
        return
    if x_arr.ndim == 2:
        if x_arr.shape[1] != reg.total_dim:
            raise ValueError(f"x has shape {x_arr.shape}, expected (N,{reg.total_dim})")
        exp = (x_arr.shape[0],) + shp
        if tuple(val.shape) != exp:
            raise ValueError(f"value shape mismatch for block '{name}': got {val.shape}, expected {exp}")
        x_arr[:, sl] = np.ascontiguousarray(val).reshape(x_arr.shape[0], -1)
        return
    raise ValueError(f"x must be 1-D or 2-D, got shape={x_arr.shape}")


# ---------------------------------------------------------------------------
# Residual Models
# ---------------------------------------------------------------------------


class ResidualModel(abc.ABC):
    """Abstract residual model interface for A/B migration."""

    @abc.abstractmethod
    def primal_block_norms(self, q: np.ndarray, z: np.ndarray, reg: BlockRegistry) -> Tuple[Dict[str, float], float]:
        """Return per-block primal residual norms and total norm."""

    @abc.abstractmethod
    def dual_block_norms(
        self, z: np.ndarray, z_prev: np.ndarray, eta: float, reg: BlockRegistry
    ) -> Tuple[Dict[str, float], float]:
        """Return per-block dual residual norms and total norm."""


class ConsensusResidual(ResidualModel):
    """Consensus ADMM residuals (B0 form).

    Primal residual per block:
        r_b = || q[:, sl_b] - z[sl_b] ||_F

    Dual residual per block:
        s_b = || eta * (z[sl_b] - z_prev[sl_b]) ||_2
    """

    def primal_block_norms(self, q: np.ndarray, z: np.ndarray, reg: BlockRegistry) -> Tuple[Dict[str, float], float]:
        q = np.asarray(q)
        z = np.asarray(z)
        if q.ndim != 2 or q.shape[1] != reg.total_dim:
            raise ValueError(f"q must have shape (N,{reg.total_dim}), got {q.shape}")
        if z.ndim != 1 or z.size != reg.total_dim:
            raise ValueError(f"z must have shape ({reg.total_dim},), got {z.shape}")

        norms: Dict[str, float] = {}
        ss = 0.0
        for name in reg.names():
            sl = reg.sl(name)
            diff = q[:, sl] - z[sl]
            n = float(np.linalg.norm(diff, ord="fro"))
            norms[name] = n
            ss += n * n
        total = float(np.sqrt(ss))
        return norms, total

    def dual_block_norms(
        self, z: np.ndarray, z_prev: np.ndarray, eta: float, reg: BlockRegistry
    ) -> Tuple[Dict[str, float], float]:
        z = np.asarray(z)
        z_prev = np.asarray(z_prev)
        if z.ndim != 1 or z.size != reg.total_dim:
            raise ValueError(f"z must have shape ({reg.total_dim},), got {z.shape}")
        if z_prev.ndim != 1 or z_prev.size != reg.total_dim:
            raise ValueError(f"z_prev must have shape ({reg.total_dim},), got {z_prev.shape}")
        if not np.isfinite(eta) or eta <= 0:
            raise ValueError(f"eta must be positive finite, got {eta}")

        norms: Dict[str, float] = {}
        ss = 0.0
        for name in reg.names():
            sl = reg.sl(name)
            diff = eta * (z[sl] - z_prev[sl])
            n = float(np.linalg.norm(diff, ord=2))
            norms[name] = n
            ss += n * n
        total = float(np.sqrt(ss))
        return norms, total


class LinearResidual(ResidualModel):
    """Placeholder for future linear residual model (A/B migration).

    TODO: implement linear form using an assembler from `problem`.
    """

    def __init__(self, problem: object, reg: BlockRegistry, flags: FeatureFlags):
        self.problem = problem
        self.reg = reg
        self.flags = flags

    def primal_block_norms(self, q: np.ndarray, z: np.ndarray, reg: BlockRegistry) -> Tuple[Dict[str, float], float]:
        raise NotImplementedError("LinearResidual is a TODO: implement when A/B linear residual form is ready.")

    def dual_block_norms(
        self, z: np.ndarray, z_prev: np.ndarray, eta: float, reg: BlockRegistry
    ) -> Tuple[Dict[str, float], float]:
        raise NotImplementedError("LinearResidual is a TODO: implement when A/B linear residual form is ready.")


def make_residual_model(problem: object, reg: BlockRegistry, flags: FeatureFlags) -> ResidualModel:
    """Residual model factory.

    Default: ConsensusResidual.
    If ``flags.use_linear_residual`` is True, returns LinearResidual (TODO).
    """

    use_linear = getattr(flags, "use_linear_residual", False)
    if bool(use_linear):
        return LinearResidual(problem, reg, flags)
    return ConsensusResidual()


# ---------------------------------------------------------------------------
# Stopping thresholds (epsilon) for ADMM
# ---------------------------------------------------------------------------


def compute_eps_pri(eps_abs: float, eps_rel: float, q: np.ndarray, z: np.ndarray) -> float:
    """Primal stopping threshold ε_pri.

    For consensus ADMM with stacked local copies q ∈ R^{N×D} and consensus z ∈ R^{D}.

    We use a stable scaling form (Boyd et al., ADMM):

        ε_pri = sqrt(N·D)·ε_abs + ε_rel · max( ||q||_F, sqrt(N)·||z||_2 )

    Notes
    -----
    * ``||q||_F`` treats q as a stacked vector in R^{N·D}.
    * ``sqrt(N)·||z||_2`` matches the scale of the stacked consensus copies.
    """

    q = np.asarray(q)
    z = np.asarray(z)
    if q.ndim != 2:
        raise ValueError(f"q must be 2-D (N,D), got shape={q.shape}")
    if z.ndim != 1:
        raise ValueError(f"z must be 1-D (D,), got shape={z.shape}")
    if q.shape[1] != z.size:
        raise ValueError(f"Dimension mismatch: q.shape[1]={q.shape[1]} vs z.size={z.size}")

    N, D = q.shape
    abs_term = np.sqrt(N * D) * float(eps_abs)
    rel_term = float(eps_rel) * max(float(np.linalg.norm(q, ord="fro")), float(np.sqrt(N) * np.linalg.norm(z, ord=2)))
    out = abs_term + rel_term
    if not np.isfinite(out) or out <= 0:
        raise FloatingPointError(f"compute_eps_pri produced invalid value: {out}")
    return float(out)


def compute_eps_dual(eps_abs: float, eps_rel: float, u: np.ndarray, eta: float) -> float:
    """Dual stopping threshold ε_dual.

    Using scaled dual variables u ∈ R^{N×D} (stacked) and penalty eta (ρ).

        ε_dual = sqrt(N·D)·ε_abs + ε_rel · ||eta·u||_F

    This matches the stacked variable protocol where u is maintained per robot.
    """

    u = np.asarray(u)
    if u.ndim != 2:
        raise ValueError(f"u must be 2-D (N,D), got shape={u.shape}")
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError(f"eta must be positive finite, got {eta}")

    N, D = u.shape
    abs_term = np.sqrt(N * D) * float(eps_abs)
    rel_term = float(eps_rel) * float(eta) * float(np.linalg.norm(u, ord="fro"))
    out = abs_term + rel_term
    if not np.isfinite(out) or out <= 0:
        raise FloatingPointError(f"compute_eps_dual produced invalid value: {out}")
    return float(out)


# ---------------------------------------------------------------------------
# Optional helpers
# ---------------------------------------------------------------------------


def decode_pos(reg: BlockRegistry, z: np.ndarray, N: int) -> np.ndarray:
    """Decode stacked position block from z to (N,2).

    Requires block 'pos' to exist with shape (2*N,).

    Flatten convention
    ------------------
    ``pos`` is stored as ``[x0, y0, x1, y1, ..., x_{N-1}, y_{N-1}]``.
    """

    if "pos" not in reg.names():
        raise KeyError("Registry does not contain 'pos' block")
    pos = get_block(reg, np.asarray(z), "pos")
    exp = (2 * N,)
    if tuple(pos.shape) != exp:
        raise ValueError(f"pos block shape mismatch: got {pos.shape}, expected {exp}")
    return pos.reshape(N, 2)
