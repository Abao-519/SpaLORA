#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
PY=/root/miniconda3/envs/SpaLORA/bin/python
ROOT=/root/night23a_working
CONTRACT=configs/night23a/stage_a_contract.json
mkdir -p "$ROOT/predictions" "$ROOT/replays" "$ROOT/evaluation" "$ROOT/stage_a_summary" "$ROOT/logs"
for lane in P22_K9 MISAR_K7 HUMAN_HIPPOCAMPUS_K7 MELANOMA_TUMOR_K2; do
  "$PY" scripts/night23a/train_loso_edge_model.py \
    --contract "$CONTRACT" --feature-dir "$ROOT/features" --teacher-dir "$ROOT/teachers" \
    --heldout "$lane" --output "$ROOT/predictions/${lane}" \
    > "$ROOT/logs/train_${lane}.log" 2>&1
  "$PY" scripts/night23a/replay_loso_edge_model.py \
    --features "$ROOT/features/${lane}.npz" \
    --prediction "$ROOT/predictions/${lane}.npz" \
    --checkpoint "$ROOT/predictions/${lane}.pt" \
    --manifest "$ROOT/predictions/${lane}.json" \
    --output "$ROOT/replays/${lane}.json" > "$ROOT/logs/replay_${lane}.log" 2>&1
  "$PY" scripts/night23a/evaluate_edge_identifiability.py \
    --prediction "$ROOT/predictions/${lane}.npz" \
    --manifest "$ROOT/predictions/${lane}.json" \
    --replay "$ROOT/replays/${lane}.json" \
    --teacher "$ROOT/teachers/${lane}.npz" \
    --output "$ROOT/evaluation/${lane}.csv" > "$ROOT/logs/evaluate_${lane}.log" 2>&1
done
"$PY" scripts/night23a/summarize_stage_a_gate.py \
  --contract "$CONTRACT" --evaluation-dir "$ROOT/evaluation" \
  --prediction-dir "$ROOT/predictions" --teacher-dir "$ROOT/teachers" \
  --output-dir "$ROOT/stage_a_summary" | tee "$ROOT/logs/stage_a_gate.log"
