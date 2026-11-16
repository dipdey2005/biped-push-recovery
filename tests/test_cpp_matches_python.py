"""Verify the C++ kernel: (1) trajectory matches the Python reference, (2) alpha filter matches RTL."""
import sys, subprocess, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from sim.models import cp_ude_no_alpha, simulate

BUILD = ROOT / "build"
csv = BUILD / "cpp_traj.csv"
subprocess.run([str(BUILD / "cp_ude_sim"), "--out", str(csv)], check=True)

data = np.genfromtxt(csv, delimiter=",", names=True)
t, L = simulate(cp_ude_no_alpha())
worst = 0.0
for k in ("x", "xdot", "xi", "xi_ref", "dhat", "dx", "px", "vx"):
    err = np.abs(data[k] - L[k]).max()
    worst = max(worst, err)
    print(f"  {k:7s} max |C++ - Python| = {err:.2e}")
assert len(data) == len(t), "length mismatch"
assert worst < 1e-9, f"C++ and Python trajectories differ by {worst:.2e}"
print(f"trajectory: PASS (worst difference {worst:.2e}, {len(t)} samples)")

r = subprocess.run([str(BUILD / "test_alpha_filter")])
assert r.returncode == 0, "alpha filter does not match RTL"
print("ALL C++ CHECKS PASSED")
