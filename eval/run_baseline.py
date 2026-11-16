"""Fixed-step RK4 simulation of the CP+UDE loop (replaces CP_UDE_alpha.slx)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sim.lip import Params, xi_ref, xi_dot_ref, disturbance
from control.cp_ude import compute_vx, solve_loop


def signals(t, s, p, ude=True):
    x, xdot, int_vx = s
    om = p.omega
    xi = x + xdot / om
    vx = compute_vx(xi_ref(t, p), xi_dot_ref(t, p), xi, p.k1)
    dx = disturbance(t, p)
    if ude:
        dhat, px, xddot = solve_loop(x, xdot, int_vx, vx, dx, om, p.tau, p.alpha)
    else:
        dhat = 0.0
        px = xi - vx / om
        xddot = om**2 * (x - px) + dx
    return dict(xi=xi, vx=vx, dx=dx, dhat=dhat, px=px, xddot=xddot)


def deriv(t, s, p, ude):
    sg = signals(t, s, p, ude)
    return np.array([s[1], sg["xddot"], sg["vx"]])


def simulate(p: Params, dt=1e-3, ude=True):
    n = int(p.T_sim / dt) + 1
    t = np.arange(n) * dt
    s = np.zeros(3)  # x0, xdot0, intvx0
    log = {k: np.zeros(n) for k in ("x", "xdot", "xi", "dhat", "dx", "px")}
    for i in range(n):
        sg = signals(t[i], s, p, ude)
        log["x"][i], log["xdot"][i] = s[0], s[1]
        for k in ("xi", "dhat", "dx", "px"):
            log[k][i] = sg[k]
        k1 = deriv(t[i], s, p, ude)
        k2 = deriv(t[i] + dt / 2, s + dt / 2 * k1, p, ude)
        k3 = deriv(t[i] + dt / 2, s + dt / 2 * k2, p, ude)
        k4 = deriv(t[i] + dt, s + dt * k3, p, ude)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return t, log


if __name__ == "__main__":
    p = Params()
    res = {}
    for ude in (True, False):
        t, L = simulate(p, ude=ude)
        res[ude] = L
        err = xi_ref(t, p) - L["xi"]
        tag = "UDE   " if ude else "no UDE"
        msg = f"{tag}: max |xi err| after 2 s = {np.abs(err[t>2]).max():.4f} m"
        if ude:
            msg += f", max |dhat-dx| after 2 s = {np.abs(L['dhat'][t>2]-L['dx'][t>2]).max():.4f}"
        print(msg)
    Lu, Ln = res[True], res[False]
    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    ax[0].plot(t, xi_ref(t, p), "k--", label="xi_ref")
    ax[0].plot(t, Lu["xi"], label="xi (UDE)")
    ax[0].plot(t, Ln["xi"], label="xi (no UDE)")
    ax[0].set_ylabel("capture point (m)"); ax[0].legend(); ax[0].grid()
    ax[1].plot(t, Lu["dx"], "k--", label="d_x")
    ax[1].plot(t, Lu["dhat"], label="d_hat")
    ax[1].set_ylabel("disturbance"); ax[1].legend(); ax[1].grid()
    ax[2].plot(t, Lu["x"], label="CoM x (UDE)")
    ax[2].plot(t, Lu["px"], label="foot p_x (UDE)")
    ax[2].set_ylabel("m"); ax[2].set_xlabel("time (s)"); ax[2].legend(); ax[2].grid()
    fig.tight_layout()
    out = pathlib.Path(__file__).parent / "plots" / "baseline.png"
    fig.savefig(out, dpi=120)
    print("saved", out)
