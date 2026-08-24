#!/usr/bin/env bash
set -euo pipefail

repo=${1:-/root/SpaLORA-night16e}
work=${2:-/root/night16e_working/frontier_v1}
python=${NIGHT16E_PYTHON:-/root/miniconda3/envs/SpaLORA/bin/python}
kit=/root/night16d_assets/kit/local_compute_kit
retained=/root/night16d_assets/retained
starts=/root/night16d_assets/starts

mkdir -p "$work"
export PYTHONPATH="$repo"

run_lane() {
  local data_id=$1
  local lane=$2
  local k=$3
  local registry=$4
  local out="$work/$lane"
  mkdir -p "$out"
  "$python" "$repo/scripts/night16e/night16e_producer.py" \
    --kit-root "$kit" --retained-root "$retained" --starts-root "$starts" \
    --data-id "$data_id" --lane "$lane" --k "$k" --registry "$registry" \
    --output "$out/partitions.npz" >"$out/producer.log" 2>&1
  "$python" "$repo/scripts/night16e/night16e_evaluator.py" \
    --kit-root "$kit" --data-id "$data_id" --k "$k" \
    --partition-bank "$out/partitions.npz" \
    --producer-json "$out/partitions.producer.json" \
    --output "$out/evaluation.csv" >"$out/evaluator.log" 2>&1
}

run_lane D1 D1 10 "$repo/configs/night16e/protein_screen_v1.json"
run_lane tonsil_s2 tonsil_s2 4 "$repo/configs/night16e/protein_screen_v1.json"
run_lane tonsil_s3 tonsil_s3 4 "$repo/configs/night16e/protein_screen_v1.json"
run_lane MISAR_E15_5_S1 MISAR_E15_5_S1 7 "$repo/configs/night16e/chromatin_screen_v1.json"

date -u +%FT%TZ >"$work/COMPLETE"
