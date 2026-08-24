#!/usr/bin/env bash
set -euo pipefail

repo=${1:-/root/SpaLORA-night16e}
work=${2:-/root/night16e_working/final_replays}
python=${NIGHT16E_PYTHON:-/root/miniconda3/envs/SpaLORA/bin/python}
export PYTHONPATH="$repo"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$work"

"$repo/scripts/night16e/run_family_frozen_replay.sh" "$repo" "$work/family_run1"
"$repo/scripts/night16e/run_family_frozen_replay.sh" "$repo" "$work/family_run2"
"$repo/scripts/night16e/run_score_frontier_replay.sh" \
  "$repo" /root/night16e_working/summary_v3/replay_registries "$work/frontier_run1"
"$repo/scripts/night16e/run_score_frontier_replay.sh" \
  "$repo" /root/night16e_working/summary_v3/replay_registries "$work/frontier_run2"

run_human() {
  local name=$1 out="$work/$1"
  mkdir -p "$out"
  "$python" "$repo/scripts/night16e/human_hippocampus_producer.py" \
    --rna /root/night16e_external/human_hippocampus/Human_RNA.h5ad \
    --atac /root/night16e_external/human_hippocampus/Human_ATAC_lsi.h5ad \
    --registry "$repo/configs/night16e/chromatin_family_frozen_v2.json" \
    --k 7 --output "$out/partitions.npz" >"$out/producer.log" 2>&1
  "$python" "$repo/scripts/night16e/human_hippocampus_evaluator.py" \
    --partition-bank "$out/partitions.npz" \
    --producer-json "$out/partitions.producer.json" \
    --reference /root/night16e_external/human_hippocampus/human_adata1_official_result.h5ad \
    --label-column true_label --k 7 --output "$out/evaluation.csv" \
    >"$out/evaluator.log" 2>&1
}
run_human human_run1
run_human human_run2

"$python" "$repo/scripts/night16e/audit_exact_replays.py" \
  --repo "$repo" --work "$work" --output "$work/exact_replay_audit.json"
date -u +%FT%TZ >"$work/COMPLETE"
