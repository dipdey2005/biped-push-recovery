// Simulate the CP+UDE loop for 10 s at 1 ms and write the trajectory as CSV.
//   usage: cp_ude_sim [--out file.csv]
#include <cstdio>
#include <cstring>
#include <cmath>
#include "cp_ude.hpp"

int main(int argc, char** argv) {
    const char* out_path = "cpp_traj.csv";
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], "--out") == 0) out_path = argv[i + 1];

    const cpude::Params p;
    const double dt = 1e-3, T = 10.0;
    const int n = static_cast<int>(std::lround(T / dt)) + 1;

    std::FILE* f = std::fopen(out_path, "w");
    if (!f) { std::perror(out_path); return 1; }
    std::fprintf(f, "t,x,xdot,xi,xi_ref,dhat,dx,px,vx\n");

    cpude::State s{0.0, 0.0, 0.0};
    for (int i = 0; i < n; ++i) {
        const double t = i * dt;
        const cpude::Signals sg = cpude::signals(t, s, p);
        std::fprintf(f, "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                     t, s[0], s[1], sg.xi, sg.xi_ref, sg.dhat, sg.dx, sg.px, sg.vx);
        s = cpude::rk4_step(t, s, dt, p);
    }
    std::fclose(f);
    std::printf("wrote %d samples to %s\n", n, out_path);
    return 0;
}
