# Biped push-recovery: capture-point control with UDE, behavior cloning, and a C++ kernel

Port of my DRDO internship MATLAB/Simulink work (LIPM + capture-point (CP) control + UDE
disturbance estimator) to Python and C++, plus an imitation-learning pipeline on top.

## Layout
| Path | What |
|---|---|
| `sim/models.py`, `sim/lip.py` | Python ports of the Simulink models (fixed-step RK4, 1 ms) |
| `control/` | CP + UDE controller functions, baseline controller as a function of the observation |
| `sim/env.py` | Gymnasium env: random pushes, sensor noise, leg-reach limit |
| `learning/` | Demo collection from the baseline, behavior cloning (PyTorch MLP) |
| `eval/` | Model plots, random-episode success rate, max recoverable push |
| `cpp/` | C++17 kernel: CP+UDE loop and a Q1.15 fixed-point alpha filter, built with CMake |
| `tests/` | C++ vs Python trajectory check; alpha filter vs Verilog RTL output |

## Run
```
pip install numpy matplotlib gymnasium
pip install torch --index-url https://download.pytorch.org/whl/cpu
bash scripts/run_all.sh      # Python models, demos, BC, evaluation
bash scripts/build_cpp.sh    # build C++ kernel and run its checks
```

## Results (200 random episodes, up to 2 pushes of +-1.5 m/s velocity jump)
| controller | success | mean tracking error | max single push (+/-) |
|---|---|---|---|
| baseline CP+UDE | 89.5% | 0.0192 m | 1.35 / 1.30 m/s |
| behavior cloning | 89.0% | 0.0188 m | 1.35 / 1.30 m/s |

BC matches the baseline because the baseline is a linear function of its observation; it cannot
exceed it. Failures come from the foot hitting the reach limit on large or stacked pushes.

## C++ verification
- Trajectory of the C++ CP+UDE loop equals the Python reference (max difference < 1e-9).
- The Q1.15 alpha filter reproduces the Verilog RTL simulation output exactly (0 LSB error, 667 samples).

## Known issues in the original Simulink models
- `CP_Control.slx`: the xi_dot_ref block has amplitude A instead of 2*pi*A (feedforward too small).
- `UDE_Model_1`: tracking is worse with the UDE on (0.068 vs 0.018 m).
