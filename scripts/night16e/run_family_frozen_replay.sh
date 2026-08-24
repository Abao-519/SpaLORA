#!/usr/bin/env bash
set -euo pipefail

repo=${1:-/root/SpaLORA-night16e}
work=${2:?output root required}
python=${NIGHT16E_PYTHON:-/root/miniconda3/envs/SpaLORA/bin/python}
kit=/root/night16d_assets/kit/local_compute_kit
retained=/root/night16d_assets/retained
starts=/root/night16d_assets/starts
export PYTHONPATH="$repo"
mkdir -p "$work"

run_lane() {
  local data_id=$1 lane=$2 k=$3 registry=$4 label_key=${5:-labels_primary}
  local out="$work/$lane"
  mkdir -p "$out"
  "$python" "$repo/scripts/night16e/night16e_producer.py" \
    --kit-root "$kit" --retained-root "$retained" --starts-root "$starts" \
    --data-id "$data_id" --lane "$lane" --k "$k" --registry "$registry" \
    --output "$out/partitions.npz" >"$out/producer.log" 2>&1
  "$python" "$repo/scripts/night16e/night16e_evaluator.py" \
    --kit-root "$kit" --data-id "$data_id" --k "$k" \
    --label-key "$label_key" --mask-key label_mask \
    --partition-bank "$out/partitions.npz" \
    --producer-json "$out/partitions.producer.json" \
    --output "$out/evaluation.csv" >"$out/evaluator.log" 2>&1
}

protein="$repo/configs/night16e/protein_family_frozen_v1.json"
chromatin="$repo/configs/night16e/chromatin_family_frozen_v2.json"
run_lane A1 A1 10 "$protein"
run_lane tonsil_s1 tonsil_s1 4 "$protein"
run_lane D1 D1 10 "$protein"
run_lane tonsil_s2 tonsil_s2 4 "$protein"
run_lane tonsil_s3 tonsil_s3 4 "$protein"
run_lane P22 P22 9 "$chromatin"
run_lane MISAR_E15_5_S1 MISAR_E15_5_S1 7 "$chromatin"
date -u +%FT%TZ >"$work/COMPLETE"
