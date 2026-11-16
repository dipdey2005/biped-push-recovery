"""Baseline CP+UDE controller written as a pure function of the observation."""
import numpy as np


def baseline_action(obs, omega, k, L_max):
    """obs = [x, xdot, xi, dhat, xi_ref, xi_dot_ref]  ->  a in [-1, 1],  px = x + L_max * a"""
    x, xd, xi, dhat, xi_ref, xi_dot_ref = obs
    vx = xi_dot_ref + k * (xi_ref - xi)
    px = xi - vx / omega + dhat / omega**2
    return float(np.clip((px - x) / L_max, -1.0, 1.0))
