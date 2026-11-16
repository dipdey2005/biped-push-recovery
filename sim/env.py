"""Gymnasium env: LIPM + capture-point tracking with pushes, noise and a leg-reach limit.

obs    = [x, xdot, xi, dhat, xi_ref, xi_dot_ref]      (x, xdot are noisy measurements)
action = a in [-1, 1]  ->  foot position px = x_meas + L_max * a   (reach limit)
step   = 10 ms control period (zero-order hold), plant integrated with RK4 at 1 ms.
dhat   = (xdot - int(vx)) / tau   (UDE estimate, same law as CP_UDE_no_alpha)
Episode ends (fall) if |xi_ref - xi| > fall_tol.
"""
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sim.models import cp_ude_no_alpha


class CPEnv(gym.Env):
    def __init__(self, cfg=None, dt=0.01, sub=10, T=5.0, L_max=0.5, noise=1e-3,
                 push_max=1.0, d_amp_max=0.4, fall_tol=0.4):
        self.c = cfg or cp_ude_no_alpha()
        self.dt, self.sub, self.T = dt, sub, T
        self.L_max, self.noise, self.push_max = L_max, noise, push_max
        self.d_amp_max, self.fall_tol = d_amp_max, fall_tol
        self.observation_space = spaces.Box(-np.inf, np.inf, (6,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)

    # ---- helpers ----
    def _ref(self, t):
        c = self.c
        ph = c.w_ref * t + self.ph_ref
        return c.A * math.sin(ph), c.xidot_amp * math.sin(ph + math.pi / 2)

    def _observe(self):
        c, om = self.c, self.c.omega
        self.xm = self.x + self.noise * self.np_random.standard_normal()
        xdm = self.xd + self.noise * self.np_random.standard_normal()
        xi = self.xm + xdm / om
        dhat = (xdm - self.I) / c.tau
        xi_ref, xi_dot_ref = self._ref(self.t)
        self.vx = xi_dot_ref + c.k1 * (xi_ref - xi)
        return np.array([self.xm, xdm, xi, dhat, xi_ref, xi_dot_ref], dtype=np.float32)

    def _xddot(self, t, x, px):
        c = self.c
        return c.omega**2 * (x - px) + self.d_amp * math.sin(c.w_dist * t + self.ph_d)

    # ---- gym API ----
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        o, rng = options or {}, self.np_random
        self.d_amp = o.get("d_amp", rng.uniform(0.0, self.d_amp_max))
        self.ph_d = o.get("ph_d", rng.uniform(0, 2 * math.pi))
        self.ph_ref = o.get("ph_ref", rng.uniform(0, 2 * math.pi))
        if "pushes" in o:
            pushes = list(o["pushes"])
        else:
            n = int(rng.integers(0, 3))
            pushes = [(float(rng.uniform(0.5, self.T - 1.0)),
                       float(rng.uniform(-self.push_max, self.push_max))) for _ in range(n)]
        self.pushes = sorted(pushes)
        self.t, self.I = 0.0, 0.0
        self.x, self.xd = self._ref(0.0)[0], 0.0       # start on the reference
        return self._observe(), {}

    def step(self, action):
        c = self.c
        a = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        px = self.xm + self.L_max * a
        while self.pushes and self.pushes[0][0] <= self.t + 1e-9:
            self.xd += self.pushes.pop(0)[1]
        h = self.dt / self.sub
        x, xd, t = self.x, self.xd, self.t
        for _ in range(self.sub):                       # RK4, px held (zero-order hold)
            k1x, k1v = xd, self._xddot(t, x, px)
            k2x, k2v = xd + h/2*k1v, self._xddot(t + h/2, x + h/2*k1x, px)
            k3x, k3v = xd + h/2*k2v, self._xddot(t + h/2, x + h/2*k2x, px)
            k4x, k4v = xd + h*k3v, self._xddot(t + h, x + h*k3x, px)
            x += h/6 * (k1x + 2*k2x + 2*k3x + k4x)
            xd += h/6 * (k1v + 2*k2v + 2*k3v + k4v)
            t += h
        self.x, self.xd, self.t = x, xd, t
        self.I += self.vx * self.dt
        obs = self._observe()
        xi_true = x + xd / c.omega
        err = self._ref(t)[0] - xi_true
        terminated = abs(err) > self.fall_tol
        truncated = t >= self.T - 1e-9
        reward = -abs(err) - 0.01 * a * a
        return obs, reward, terminated, truncated, {"err": err}
