#!/usr/bin/env bash
# Reproduce everything: Simulink-model ports, demos, behavior cloning, evaluation.
set -e
cd "$(dirname "$0")/.."
python eval/run_models.py
python learning/collect_demos.py
python learning/bc.py
python eval/eval_policy.py
