#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
PY=/root/miniconda3/envs/SpaLORA/bin/python
ROOT=/root/night23a_working
CONTRACT=configs/night23a/stage_a_contract.json
mkdir -p "$ROOT/features" "$ROOT/teachers" "$ROOT/logs"
for lane in P22_K9 MISAR_K7 HUMAN_HIPPOCAMPUS_K7 MELANOMA_TUMOR_K2; do
  "$PY" scripts/night23a/build_edge_features.py \
    --contract "$CONTRACT" --lane "$lane" --output "$ROOT/features/${lane}.npz" \
    > "$ROOT/logs/features_${lane}.log" 2>&1
  "$PY" scripts/night23a/build_teacher_relation.py \
    --contract "$CONTRACT" --lane "$lane" --features "$ROOT/features/${lane}.npz" \
    --output "$ROOT/teachers/${lane}.npz" > "$ROOT/logs/teacher_${lane}.log" 2>&1
done
