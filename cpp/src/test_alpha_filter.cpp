// Check the C++ Q1.15 alpha filter against the Verilog RTL simulation output (bit-exact).
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include "alpha_filter.hpp"

#ifndef TESTDATA_DIR
#define TESTDATA_DIR "testdata"
#endif

int main() {
    std::ifstream fin(std::string(TESTDATA_DIR) + "/test_input.hex");
    std::ifstream fout(std::string(TESTDATA_DIR) + "/rtl_output.txt");
    if (!fin || !fout) { std::fprintf(stderr, "cannot open test vectors in %s\n", TESTDATA_DIR); return 2; }

    std::vector<int32_t> in, expected;
    std::string line;
    while (std::getline(fin, line)) {
        if (line.find_first_of("0123456789abcdefABCDEF") == std::string::npos) continue;
        int32_t v = static_cast<int32_t>(std::stoul(line, nullptr, 16));
        if (v >= 32768) v -= 65536;                       // 16-bit two's complement
        in.push_back(v);
    }
    int v;
    while (fout >> v) expected.push_back(v);
    if (in.size() != expected.size() || in.empty()) {
        std::fprintf(stderr, "length mismatch: %zu inputs, %zu outputs\n", in.size(), expected.size());
        return 2;
    }

    AlphaFilterQ15 filt;
    int mismatches = 0, max_err = 0;
    for (size_t i = 0; i < in.size(); ++i) {
        const int y = filt.step(in[i]);
        const int err = std::abs(y - expected[i]);
        if (err) { ++mismatches; if (err > max_err) max_err = err; }
    }
    std::printf("alpha filter vs RTL: %zu samples, %d mismatches, max error %d LSB -> %s\n",
                in.size(), mismatches, max_err, mismatches ? "FAIL" : "PASS");
    return mismatches ? 1 : 0;
}
