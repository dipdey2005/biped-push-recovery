#!/usr/bin/env bash
# Build the C++ kernel and run its checks against the Python reference and the RTL vectors.
set -e
cd "$(dirname "$0")/.."
cmake -S cpp -B build
cmake --build build -j
python tests/test_cpp_matches_python.py
