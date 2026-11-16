// First-order alpha low-pass filter in Q1.15 fixed point, bit-exact with the Verilog RTL.
//   y[n] = (B0*x[n] + B1*x[n-1] - A1*y[n-1]) >> 15
// Coefficients from the bilinear design in runtest.m. The 64-bit accumulator cannot overflow;
// '>>' on a negative signed value is an arithmetic shift (floor) on GCC/Clang, as in the RTL.
#pragma once
#include <cstdint>

struct AlphaFilterQ15 {
    static constexpr int32_t B0 = 799;
    static constexpr int32_t B1 = 799;
    static constexpr int32_t A1 = -31170;
    static constexpr int Q = 15;

    int32_t x_prev = 0;
    int32_t y_prev = 0;

    int32_t step(int32_t x) {
        const int64_t acc = static_cast<int64_t>(B0) * x
                          + static_cast<int64_t>(B1) * x_prev
                          - static_cast<int64_t>(A1) * y_prev;
        const int32_t y = static_cast<int32_t>(acc >> Q);
        x_prev = x;
        y_prev = y;
        return y;
    }
};
