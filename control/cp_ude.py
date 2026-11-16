"""Capture-point controller with UDE alpha-filter (port of compute_*.m)."""
import numpy as np


def compute_vx(xi_ref, xi_dot_ref, xi, k):
    return xi_dot_ref + k * (xi_ref - xi)


def compute_px(vx, dhat, xi, om):
    return xi - vx / om + dhat / om**2


def compute_dhat_alpha(xddot, int_vx, tau, alpha, vx, xdot):
    return ((1 - alpha) / alpha) * (xddot - vx) + (1 / (alpha * tau)) * (xdot - int_vx)


def solve_loop(x, xdot, int_vx, vx, dx, om, tau, alpha):
    """Simulink solves an algebraic loop here: dhat -> px -> xddot -> dhat.
    The loop is linear in dhat, so solve it in closed form.
    (Plain fixed-point iteration diverges: loop gain = -(1-alpha)/alpha.)
    Uses xddot = -om*xdot + om*vx - dhat + dx  (from plant + px, with xi = x + xdot/om).
    """
    dhat = (1 - alpha) * (dx - om * xdot + (om - 1) * vx) + (xdot - int_vx) / tau
    xi = x + xdot / om
    px = compute_px(vx, dhat, xi, om)
    xddot = om**2 * (x - px) + dx
    return dhat, px, xddot


# ---- functions from LIPM_Model_1 / UDE_Model_1 (no alpha-filter) ----
def compute_xi(x, xdot, om):
    """Capture point: xi = x + xdot/omega"""
    return x + xdot / om


def compute_dhat(xdot, int_vx, tau):
    """UDE estimate (no alpha): dhat = (xdot - int(vx)) / tau"""
    return (xdot - int_vx) / tau


def compute_px_ude(vx, dhat, xi, om):
    """UDE_Model_1 law:  px = xi - (vx - dhat)/omega"""
    return xi - (vx - dhat) / om
