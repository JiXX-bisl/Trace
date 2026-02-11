# scripts/core/inner/residual_balancing.py
from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np

from scripts.core.data import FeatureFlags


def residual_balance_step(
    eta: float,
    u: np.ndarray,
    r_norm: float,
    s_norm: float,
    params: Any,
    flags: FeatureFlags,
) -> Tuple[float, np.ndarray, Dict[str, Any]]:
    """
    Scaled ADMM residual balancing.

    Rule:
      if r > mu * s:      eta <- eta * tau_incr ; u <- u * (eta_old / eta_new)
      elif s > mu * r:    eta <- eta / tau_decr ; u <- u * (eta_old / eta_new)
      else: unchanged

    Optional clamps:
      eta_min / eta_max (from params or flags)

    Returns:
      (eta_new, u_scaled, info)
    """
    if not bool(getattr(flags, "enable_residual_balancing", False)):
        return float(eta), u, {"changed": False}

    if eta <= 0:
        raise ValueError(f"eta must be positive, got {eta}")

    mu = float(getattr(params, "mu", 10.0))
    tau_incr = float(getattr(params, "tau_incr", 2.0))
    tau_decr = float(getattr(params, "tau_decr", 2.0))

    eta_min = float(getattr(params, "eta_min", getattr(flags, "eta_min", 1e-6)))
    eta_max = float(getattr(params, "eta_max", getattr(flags, "eta_max", 1e6)))

    eta_old = float(eta)
    eta_new = eta_old
    direction = "none"

    if r_norm > mu * s_norm:
        eta_new = eta_old * tau_incr
        direction = "incr"
    elif s_norm > mu * r_norm:
        eta_new = eta_old / tau_decr
        direction = "decr"

    eta_new = float(np.clip(eta_new, eta_min, eta_max))

    if eta_new == eta_old:
        return eta_old, u, {"changed": False, "direction": "none", "ratio": 1.0}

    # keep unscaled dual y = eta*u constant -> u <- u * (eta_old/eta_new)
    ratio = eta_old / eta_new
    u_scaled = u * ratio

    info = {"changed": True, "direction": direction, "eta_old": eta_old, "eta_new": eta_new, "ratio": ratio}
    return eta_new, u_scaled, info
