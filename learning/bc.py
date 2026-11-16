"""Behavior cloning: small MLP imitating the baseline controller (PyTorch, CPU)."""
import sys, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn as nn

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/demos.npz")
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--out", default="models/bc.pt")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed)
d = np.load(args.data)
X, Y = d["obs"], d["act"][:, None]
mu, sd = X.mean(0), X.std(0) + 1e-8
Xn = (X - mu) / sd
idx = np.random.permutation(len(X)); nv = len(X) // 10
va, tr = idx[:nv], idx[nv:]
Xt, Yt = torch.tensor(Xn[tr]), torch.tensor(Y[tr])
Xv, Yv = torch.tensor(Xn[va]), torch.tensor(Y[va])

net = nn.Sequential(nn.Linear(6, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
for ep in range(1, args.epochs + 1):
    perm = torch.randperm(len(Xt))
    for i in range(0, len(Xt), 256):
        b = perm[i:i + 256]
        loss = nn.functional.mse_loss(net(Xt[b]), Yt[b])
        opt.zero_grad(); loss.backward(); opt.step()
    if ep % 10 == 0 or ep == 1:
        with torch.no_grad():
            print(f"epoch {ep:3d}  val MSE {nn.functional.mse_loss(net(Xv), Yv).item():.6f}")
pathlib.Path(args.out).parent.mkdir(exist_ok=True)
torch.save({"state": net.state_dict(), "mu": mu, "sd": sd}, args.out)
print("saved", args.out)
