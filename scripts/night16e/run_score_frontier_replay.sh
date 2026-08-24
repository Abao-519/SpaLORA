#!/usr/bin/env bash
set -euo pipefail

repo=${1:-/root/SpaLORA-night16e}
registry_root=${2:-/root/night16e_working/summary_v2/replay_registries}
work=${3:?output root required}
python=${NIGHT16E_PYTHON:-/root/miniconda3/envs/SpaLORA/bin/python}
kit=/root/night16d_assets/kit/local_compute_kit
retained=/root/night16d_assets/retained
starts=/root/night16d_assets/starts
export PYTHONPATH="$repo"
mkdir -p "$work"

run_lane() {
  local data_id=$1 lane=$2 k=$3
  local out="$work/$lane"
  mkdir -p "$out"
  "$python" "$repo/scripts/night16e/night16e_producer.py" \
    --kit-root "$kit" --retained-root "$retained" --starts-root "$starts" \
    --data-id "$data_id" --lane "$lane" --k "$k" \
    --registry "$registry_root/$lane.json" --output "$out/partitions.npz" \
    >"$out/producer.log" 2>&1
  "$python" "$repo/scripts/night16e/night16e_evaluator.py" \
    --kit-root "$kit" --data-id "$data_id" --k "$k" \
    --partition-bank "$out/partitions.npz" \
    --producer-json "$out/partitions.producer.json" --output "$out/evaluation.csv" \
    >"$out/evaluator.log" 2>&1
}

run_lane A1 A1 10
run_lane tonsil_s1 tonsil_s1 4
run_lane D1 D1 10
run_lane tonsil_s2 tonsil_s2 4
run_lane tonsil_s3 tonsil_s3 4
run_lane P22 P22 9
run_lane MISAR_E15_5_S1 MISAR_E15_5_S1 7
date -u +%FT%TZ >"$work/COMPLETE"
