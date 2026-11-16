"""The three Simulink models, as fixed-step (RK4, 1 ms = Simulink ode4) simulations.

  cp_control()       LIPM_Model_1/CP_Control.slx         plain CP, no UDE, no disturbance
  cp_ude_no_alpha()  LIPM_Model_1/CP_UDE_no_alpha.slx    CP + UDE (px = xi - vx/w + dhat/w^2)
  ude_model_1()      UDE_Model_1/build_CP_UDE_model.m    CP + UDE (px = xi - (vx - dhat)/w)

States: s = [x, xdot, int_vx].   Block wiring is the same in all three:
  x, xdot -> xi -> vx -> (int_vx) -> dhat -> px -> plant -> xddot -> xdot -> x
"""
from dataclasses import dataclass
import numpy as np
from control.cp_ude import compute_xi, compute_vx, compute_dhat, compute_px, compute_px_ude


@dataclass
class Cfg:
    name: str
    law: str                 # 'plain' | 'dhat_w2' | 'dhat_w'
    tau: float
    A: float                 # xi_ref amplitude
    w_ref: float             # xi_ref frequency (rad/s)
    xidot_amp: float         # amplitude of the xi_dot_ref sine block (as wired in Simulink)
    d_amp: float             # disturbance amplitude (0 = no disturbance input)
    w_dist: float            # disturbance frequency (rad/s)
    k1: float = 3.0
    g: float = 9.81
    zc: float = 0.81
    T_sim: float = 10.0

    @property
    def omega(self):
        return float(np.sqrt(self.g / self.zc))


def cp_control():
    A = 0.1
    # NOTE: in CP_Control.slx the xi_dot_ref block has amplitude A (not A*2*pi) at 2*pi rad/s
    return Cfg("CP_Control", "plain", tau=1.0, A=A, w_ref=2*np.pi, xidot_amp=A, d_amp=0.0, w_dist=2*np.pi)


def cp_ude_no_alpha():
    A = 0.1
    return Cfg("CP_UDE_no_alpha", "dhat_w2", tau=0.5, A=A, w_ref=2*np.pi, xidot_amp=A*2*np.pi,
               d_amp=0.4, w_dist=2*np.pi)


def ude_model_1():
    A = 0.1
    # Sine Wave blocks use rad/s: Frequency '1' = 1 rad/s
    return Cfg("UDE_Model_1", "dhat_w", tau=0.05, A=A, w_ref=1.0, xidot_amp=A,
               d_amp=0.2, w_dist=1.0)


def signals(t, s, c: Cfg):
    x, xdot, int_vx = s
    om = c.omega
    xi_ref = c.A * np.sin(c.w_ref * t)
    xi_dot_ref = c.xidot_amp * np.sin(c.w_ref * t + np.pi / 2)
    dx = c.d_amp * np.sin(c.w_dist * t)
    xi = compute_xi(x, xdot, om)
    vx = compute_vx(xi_ref, xi_dot_ref, xi, c.k1)
    if c.law == "plain":
        dhat = 0.0
        px = xi - vx / om
    else:
        dhat = compute_dhat(xdot, int_vx, c.tau)
        px = compute_px(vx, dhat, xi, om) if c.law == "dhat_w2" else compute_px_ude(vx, dhat, xi, om)
    xddot = om**2 * (x - px) + dx
    return dict(xi=xi, xi_ref=xi_ref, vx=vx, dx=dx, dhat=dhat, px=px, xddot=xddot)


def _f(t, s, c):
    sg = signals(t, s, c)
    return np.array([s[1], sg["xddot"], sg["vx"]])


def simulate(c: Cfg, dt=1e-3):
    n = int(round(c.T_sim / dt)) + 1
    t = np.arange(n) * dt
    s = np.zeros(3)
    keys = ("x", "xdot", "xi", "xi_ref", "dhat", "dx", "px", "vx")
    log = {k: np.zeros(n) for k in keys}
    for i in range(n):
        sg = signals(t[i], s, c)
        log["x"][i], log["xdot"][i] = s[0], s[1]
        for k in keys[2:]:
            log[k][i] = sg[k]
        k1 = _f(t[i], s, c)
        k2 = _f(t[i] + dt/2, s + dt/2*k1, c)
        k3 = _f(t[i] + dt/2, s + dt/2*k2, c)
        k4 = _f(t[i] + dt, s + dt*k3, c)
        s = s + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    return t, log
