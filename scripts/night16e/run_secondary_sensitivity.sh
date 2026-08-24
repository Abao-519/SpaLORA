#!/usr/bin/env bash
set -euo pipefail

repo=${1:-/root/SpaLORA-night16e}
work=${2:-/root/night16e_working/secondary_sensitivity_v1}
python=${NIGHT16E_PYTHON:-/root/miniconda3/envs/SpaLORA/bin/python}
kit=/root/night16d_assets/kit/local_compute_kit
retained=/root/night16d_assets/retained
starts=/root/night16e_assets/starts
registry="$repo/configs/night16e/chromatin_screen_v1.json"

mkdir -p "$work"
export PYTHONPATH="$repo"

run_lane() {
  local data_id=$1 lane=$2 k=$3 label_key=$4
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

# Secondary protocols do not count as independent studies and cannot select
# either family-frozen headline configuration.
run_lane P22 P22_3DOT_K18 18 labels_k18_author_assignment
run_lane MISAR_E15_5_S1 MISAR_E15_5_S1_K12 12 labels_primary

date -u +%FT%TZ >"$work/COMPLETE"
