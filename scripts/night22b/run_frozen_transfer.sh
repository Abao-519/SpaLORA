#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PY=/root/miniconda3/envs/SpaLORA/bin/python
ROOT=/root/night22b_working
CONTRACT=configs/night22b/frozen_transfer_contract.json
mkdir -p "$ROOT/junction" "$ROOT/logs"

run_one() {
  local lane="$1"
  local k="$2"
  local carrier="$3"
  local start="$4"
  "$PY" scripts/night22a/night22a_junction_producer.py \
    --lane "$lane" \
    --k "$k" \
    --carrier "$carrier" \
    --embedding "$ROOT/authority/${lane}__RETAINED_CARRIER.npz" \
    --representation-source RETAINED_CARRIER \
    --parent-bank "$ROOT/authority/${lane}__TRANSFER_PARENT_AUTHORITY.npz" \
    --start-bank "$ROOT/starts/${lane}.npz" \
    --start-candidate "$start" \
    --registry "$CONTRACT" \
    --seed 0 \
    --output "$ROOT/junction/${lane}__${start}.npz" \
    > "$ROOT/logs/junction_${lane}__${start}.log" 2>&1
}

# The frozen Leiden generator has no exact-K solution on MISAR under the
# preregistered resolution grid. Only the preregistered sensitivities run.
run_one MISAR_K7 7 /root/night16f_working/carriers/MISAR_K7.npz GEOM_FEATURE_NCUT_K24
run_one MISAR_K7 7 /root/night16f_working/carriers/MISAR_K7.npz NIGHT16H_FROZEN_SELECTOR

run_one HUMAN_HIPPOCAMPUS_K7 7 /root/night16f_working/carriers/HUMAN_HIPPOCAMPUS_K7.npz GEOM_LEIDEN_FEATURE
run_one HUMAN_HIPPOCAMPUS_K7 7 /root/night16f_working/carriers/HUMAN_HIPPOCAMPUS_K7.npz GEOM_FEATURE_NCUT_K24
run_one HUMAN_HIPPOCAMPUS_K7 7 /root/night16f_working/carriers/HUMAN_HIPPOCAMPUS_K7.npz NIGHT16H_FROZEN_SELECTOR

run_one MELANOMA_TUMOR_K2 2 /root/night16f_working/carriers/MELANOMA_TUMOR_K2.npz GEOM_LEIDEN_FEATURE
run_one MELANOMA_TUMOR_K2 2 /root/night16f_working/carriers/MELANOMA_TUMOR_K2.npz GEOM_FEATURE_NCUT_K24
run_one MELANOMA_TUMOR_K2 2 /root/night16f_working/carriers/MELANOMA_TUMOR_K2.npz NIGHT16H_FROZEN_SELECTOR
