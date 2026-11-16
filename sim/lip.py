"""LIPM plant + parameters (port of init.m and compute_xddot.m)."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Params:
    g: float = 9.81
    zc: float = 0.81        # CoM height (m)
    k1: float = 3.0         # CP tracking gain
    tau: float = 0.01       # UDE time constant
    alpha: float = 0.015    # alpha-filter parameter (init.m value)
    A: float = 1.0          # xi_ref amplitude
    A0: float = 0.4         # disturbance amplitude
    w_sig: float = 2 * np.pi  # reference / disturbance frequency (rad/s)
    T_sim: float = 10.0

    @property
    def omega(self) -> float:
        return float(np.sqrt(self.g / self.zc))


def compute_xddot(x, px, om, dx):
    """LIPM plant:  xddot = omega^2 (x - px) + dx"""
    return om**2 * (x - px) + dx


def xi_ref(t, p: Params):
    return p.A * np.sin(p.w_sig * t)


def xi_dot_ref(t, p: Params):
    return p.A * p.w_sig * np.sin(p.w_sig * t + np.pi / 2)


def disturbance(t, p: Params):
    return p.A0 * np.sin(p.w_sig * t)
