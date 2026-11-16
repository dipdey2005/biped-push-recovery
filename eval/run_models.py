"""Run the three Simulink models in Python; print metrics and save scope-style plots."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import replace
from sim.models import cp_control, cp_ude_no_alpha, ude_model_1, simulate

OUT = pathlib.Path(__file__).parent / "plots"
OUT.mkdir(exist_ok=True)


def metrics(t, L, tmin=2.0):
    m = t > tmin
    return (np.abs(L["xi_ref"] - L["xi"])[m].max(),
            np.abs(L["dhat"] - L["dx"])[m].max(),
            np.abs(L["x"]).max())


for make in (cp_control, cp_ude_no_alpha, ude_model_1):
    c = make()
    t, L = simulate(c)
    t2, L2 = simulate(c, dt=2.5e-4)  # step-size check
    e, d, xm = metrics(t, L)
    drift = np.abs(L["xi"] - L2["xi"][::4]).max()
    print(f"{c.name:16s} max|xi_ref-xi|(t>2s)={e:.4f}  max|dhat-dx|={d:.4f}  max|x|={xm:.3f}  "
          f"(dt 1ms vs 0.25ms: {drift:.1e})")
    if c.law != "plain":
        t0, L0 = simulate(replace(c, law="plain"))
        print(f"{'':16s} same model with UDE off: max|xi_ref-xi|={metrics(t0, L0)[0]:.4f}")

    fig, ax = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    ax[0].plot(t, L["xi_ref"], "k--", label="xi_ref"); ax[0].plot(t, L["xi"], label="xi")
    ax[0].set_ylabel("CP_Scope"); ax[0].legend(); ax[0].grid()
    ax[1].plot(t, L["x"]); ax[1].set_ylabel("COM_Scope  x (m)"); ax[1].grid()
    ax[2].plot(t, L["px"]); ax[2].set_ylabel("px_Scope  p_x (m)"); ax[2].grid()
    if c.law != "plain":
        ax[3].plot(t, L["dx"], "k--", label="d_x"); ax[3].plot(t, L["dhat"], label="d_hat")
        ax[3].legend()
    ax[3].set_ylabel("d_hat vs dx"); ax[3].set_xlabel("time (s)"); ax[3].grid()
    fig.suptitle(c.name); fig.tight_layout()
    fig.savefig(OUT / f"{c.name}.png", dpi=110); plt.close(fig)
print("plots saved in", OUT)
