"""Roll out the baseline CP+UDE controller and log (obs, action) pairs for behavior cloning.

--act-noise adds noise to the EXECUTED action while the LABEL stays the clean baseline action
(DART-style), so the demos also cover states a slightly imperfect policy would drift into.
"""
import sys, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from sim.env import CPEnv
from control.baseline import baseline_action

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=300)
ap.add_argument("--act-noise", type=float, default=0.1)
ap.add_argument("--push-max", type=float, default=1.2)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="data/demos.npz")
args = ap.parse_args()

env = CPEnv(push_max=args.push_max)
om, k, L = env.c.omega, env.c.k1, env.L_max
rng = np.random.default_rng(args.seed)
O, A, falls = [], [], 0
for ep in range(args.episodes):
    obs, _ = env.reset(seed=args.seed * 100000 + ep)
    while True:
        a_label = baseline_action(obs, om, k, L)
        O.append(obs.copy()); A.append(a_label)
        a_exec = np.clip(a_label + args.act_noise * rng.standard_normal(), -1, 1)
        obs, r, term, trunc, _ = env.step(a_exec)
        if term or trunc:
            falls += int(term); break
O, A = np.array(O, np.float32), np.array(A, np.float32)
pathlib.Path(args.out).parent.mkdir(exist_ok=True)
np.savez(args.out, obs=O, act=A)
print(f"saved {len(O)} samples from {args.episodes} episodes ({falls} falls) -> {args.out}")
print(f"action stats: mean {A.mean():.3f}  std {A.std():.3f}  saturated {np.mean(np.abs(A) > 0.999):.1%}")
