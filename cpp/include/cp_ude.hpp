// Capture-point controller + UDE disturbance estimator on the LIPM plant (CP_UDE_no_alpha model).
// Fixed-step RK4, same equations and evaluation order as the Python reference (sim/models.py).
#pragma once
#include <array>
#include <cmath>

namespace cpude {

constexpr double kPi = 3.14159265358979323846;

struct Params {
    double g = 9.81, zc = 0.81, k1 = 3.0, tau = 0.5;
    double A = 0.1, w_ref = 2.0 * kPi, xidot_amp = 0.1 * 2.0 * kPi;  // xi_ref amplitude / freq / feedforward amplitude
    double d_amp = 0.4, w_dist = 2.0 * kPi;                            // disturbance
    double omega() const { return std::sqrt(g / zc); }
};

using State = std::array<double, 3>;  // x, xdot, int(vx)

struct Signals { double xi, xi_ref, vx, dx, dhat, px, xddot; };

inline Signals signals(double t, const State& s, const Params& p) {
    const double om = p.omega();
    const double xi = s[0] + s[1] / om;                                  // compute_xi
    const double xi_ref = p.A * std::sin(p.w_ref * t);
    const double xi_dot_ref = p.xidot_amp * std::sin(p.w_ref * t + kPi / 2.0);
    const double dx = p.d_amp * std::sin(p.w_dist * t);
    const double vx = xi_dot_ref + p.k1 * (xi_ref - xi);                 // compute_vx
    const double dhat = (s[1] - s[2]) / p.tau;                           // compute_dhat (UDE)
    const double px = xi - vx / om + dhat / (om * om);                   // compute_px
    const double xddot = om * om * (s[0] - px) + dx;                     // LIPM plant
    return {xi, xi_ref, vx, dx, dhat, px, xddot};
}

inline State deriv(double t, const State& s, const Params& p) {
    const Signals sg = signals(t, s, p);
    return {s[1], sg.xddot, sg.vx};
}

inline State rk4_step(double t, const State& s, double dt, const Params& p) {
    auto axpy = [](const State& a, double h, const State& k) {
        return State{a[0] + h * k[0], a[1] + h * k[1], a[2] + h * k[2]};
    };
    const State k1 = deriv(t, s, p);
    const State k2 = deriv(t + dt / 2, axpy(s, dt / 2, k1), p);
    const State k3 = deriv(t + dt / 2, axpy(s, dt / 2, k2), p);
    const State k4 = deriv(t + dt, axpy(s, dt, k3), p);
    State out{};
    for (int i = 0; i < 3; ++i)
        out[i] = s[i] + dt / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
    return out;
}

}  // namespace cpude
