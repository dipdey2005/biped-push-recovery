"""Compare controllers: random-episode success, tracking error, and max recoverable push."""
import sys, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sim.env import CPEnv
from control.baseline import baseline_action


def make_policy(name, env):
    om, k, L = env.c.omega, env.c.k1, env.L_max
    if name == "baseline":
        return lambda o: baseline_action(o, om, k, L)
    if name == "bc":
        import torch, torch.nn as nn
        ck = torch.load("models/bc.pt", weights_only=False)
        net = nn.Sequential(nn.Linear(6, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
        net.load_state_dict(ck["state"]); net.eval()
        mu, sd = ck["mu"], ck["sd"]
        def pol(o):
            with torch.no_grad():
                return float(net(torch.tensor(((o - mu) / sd).astype(np.float32))).item())
        return pol
    raise ValueError(name)


def run(env, pol, seed, options=None):
    obs, _ = env.reset(seed=seed, options=options)
    errs = []
    while True:
        obs, r, term, trunc, info = env.step(pol(obs))
        errs.append(abs(info["err"]))
        if term or trunc:
            return term, float(np.mean(errs))


def push_sweep(env, pol, dvs, sign):
    ok = []
    for dv in dvs:
        opt = dict(d_amp=0.4, ph_d=0.0, ph_ref=0.0, pushes=[(1.0, sign * dv)])
        fell, _ = run(env, pol, seed=1, options=opt)
        ok.append(not fell)
    return np.array(ok)


def max_recoverable(dvs, ok):
    bad = np.where(~ok)[0]
    return dvs[-1] if len(bad) == 0 else (dvs[bad[0] - 1] if bad[0] > 0 else 0.0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--controllers", nargs="+", default=["baseline", "bc"])
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--push-max", type=float, default=1.5)
    args = ap.parse_args()

    dvs = np.round(np.arange(0.0, 2.51, 0.05), 2)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    print(f"{'controller':10s} {'success':>8s} {'mean|err|(m)':>13s} {'max push +':>11s} {'max push -':>11s}   (push = velocity jump, m/s)")
    for name in args.controllers:
        env = CPEnv(push_max=args.push_max)
        pol = make_policy(name, env)
        res = [run(env, pol, seed=10_000 + i) for i in range(args.episodes)]
        succ = 1 - np.mean([r[0] for r in res]); err = np.mean([r[1] for r in res])
        okp, okn = push_sweep(env, pol, dvs, +1), push_sweep(env, pol, dvs, -1)
        print(f"{name:10s} {succ:8.1%} {err:13.4f} {max_recoverable(dvs, okp):11.2f} {max_recoverable(dvs, okn):11.2f}")
        ax.plot(dvs, okp.astype(int) + 0.02 * list(args.controllers).index(name), "o-", ms=3, label=name)
    ax.set_xlabel("single push at t = 1 s (velocity jump, m/s)"); ax.set_ylabel("recovered (1) / fell (0)")
    ax.legend(); ax.grid(); fig.tight_layout()
    out = pathlib.Path(__file__).parent / "plots" / "push_sweep.png"
    fig.savefig(out, dpi=110); print("saved", out)
