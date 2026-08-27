#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
PY=/root/miniconda3/envs/SpaLORA/bin/python
ROOT=/root/night22b_working
mkdir -p "$ROOT/junction_replays_round2"

replay_one() {
  local lane="$1" carrier="$2" start="$3"
  local stem="${lane}__${start}"
  "$PY" scripts/night22a/night22a_junction_replay.py \
    --bank "$ROOT/junction/${stem}.npz" \
    --carrier "$carrier" \
    --embedding "$ROOT/authority/${lane}__RETAINED_CARRIER.npz" \
    --output "$ROOT/junction_replays_round2/${stem}.json"
}

replay_one MISAR_K7 /root/night16f_working/carriers/MISAR_K7.npz GEOM_FEATURE_NCUT_K24
replay_one MISAR_K7 /root/night16f_working/carriers/MISAR_K7.npz NIGHT16H_FROZEN_SELECTOR
replay_one HUMAN_HIPPOCAMPUS_K7 /root/night16f_working/carriers/HUMAN_HIPPOCAMPUS_K7.npz GEOM_LEIDEN_FEATURE
replay_one HUMAN_HIPPOCAMPUS_K7 /root/night16f_working/carriers/HUMAN_HIPPOCAMPUS_K7.npz GEOM_FEATURE_NCUT_K24
replay_one HUMAN_HIPPOCAMPUS_K7 /root/night16f_working/carriers/HUMAN_HIPPOCAMPUS_K7.npz NIGHT16H_FROZEN_SELECTOR
replay_one MELANOMA_TUMOR_K2 /root/night16f_working/carriers/MELANOMA_TUMOR_K2.npz GEOM_LEIDEN_FEATURE
replay_one MELANOMA_TUMOR_K2 /root/night16f_working/carriers/MELANOMA_TUMOR_K2.npz GEOM_FEATURE_NCUT_K24
replay_one MELANOMA_TUMOR_K2 /root/night16f_working/carriers/MELANOMA_TUMOR_K2.npz NIGHT16H_FROZEN_SELECTOR
