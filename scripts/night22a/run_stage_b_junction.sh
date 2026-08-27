#!/usr/bin/env bash
set -euo pipefail
cd /root/SpaLORA-night16h
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0
export PYTHONPATH=/root/SpaLORA-night16h
PY=/root/miniconda3/envs/SpaLORA/bin/python
REG=configs/night22a/stage_b_junction_freeze.json
OUT=/root/night22a_working/junction
EVAL=/root/night22a_working/junction_evaluations
REPLAY=/root/night22a_working/junction_replays
LOG=/root/night22a_working/logs
mkdir -p "$OUT" "$EVAL" "$REPLAY" "$LOG"

run_one() {
  local lane="$1" k="$2" carrier="$3" embedding="$4" parent="$5" start="$6" authority="$7" reference="${8:-}"
  local stem="${lane}__RETAINED_CARRIER__${start}"
  if [[ ! -f "$OUT/${stem}.npz" ]]; then
    "$PY" scripts/night22a/night22a_junction_producer.py \
      --lane "$lane" --k "$k" --carrier "$carrier" --embedding "$embedding" \
      --representation-source RETAINED_CARRIER --parent-bank "$parent" \
      --start-bank "/root/night22a_working/geometry/${lane}__RETAINED_CARRIER.npz" \
      --start-candidate "$start" --registry "$REG" --seed 0 \
      --output "$OUT/${stem}.npz" > "$LOG/${stem}.log" 2>&1
  fi
  local eval_args=(--bank "$OUT/${stem}.npz" --authority "$authority" --output "$EVAL/${stem}.csv")
  if [[ -n "$reference" ]]; then eval_args+=(--reference-h5ad "$reference"); fi
  "$PY" scripts/night22a/night22a_geometry_evaluator.py "${eval_args[@]}"
  "$PY" scripts/night22a/night22a_junction_replay.py \
    --bank "$OUT/${stem}.npz" --carrier "$carrier" --embedding "$embedding" \
    --output "$REPLAY/${stem}.json"
}

for start in GEOM_FEATURE_NCUT_K24 GEOM_LEIDEN_FEATURE; do
  run_one A1_K10 10 \
    /root/night18a_working/carriers/A1_K10.npz \
    /root/night21b_working/discovery/A1_K10__T0_STRONG_CARRIER__STABLE700_V1_S0.npz \
    /root/night21c_working/endpoint_banks/A1_K10__RETAINED_CARRIER.npz "$start" \
    /root/night16d_assets/kit/local_compute_kit/A1.npz
  run_one TONSIL_S1_K4 4 \
    /root/night21a_working/carriers/TONSIL_S1_K4.npz \
    /root/night21b_working/discovery/TONSIL_S1_K4__T0_STRONG_CARRIER__STABLE700_V1_S0.npz \
    /root/night21c_working/endpoint_banks/TONSIL_S1_K4__RETAINED_CARRIER.npz "$start" \
    /root/night16d_assets/kit/local_compute_kit/tonsil_s1.npz
  run_one P22_K9 9 \
    /root/night18a_working/carriers/P22_K9.npz \
    /root/night21b_working/discovery/P22_K9__T0_STRONG_CARRIER__STABLE700_V1_S0.npz \
    /root/night21c_working/endpoint_banks/P22_K9__RETAINED_CARRIER.npz "$start" \
    /root/night16d_assets/kit/local_compute_kit/P22.npz
  run_one PLACENTA_K10 10 \
    /root/night18d_working/carrier/placenta_carrier.npz \
    /root/night21b_working/discovery/PLACENTA_K10__T0_STRONG_CARRIER__STABLE700_V1_S0.npz \
    /root/night21c_working/endpoint_banks/PLACENTA_K10__RETAINED_CARRIER.npz "$start" \
    /root/night18d_working/carrier/placenta_carrier.npz \
    "/root/autodl-fs/Human placenta architecture/humanplacenta_rna.h5ad"
done

echo "JUNCTION_BANKS=$(find "$OUT" -maxdepth 1 -name '*.npz' | wc -l)"
echo "JUNCTION_EVALUATIONS=$(find "$EVAL" -maxdepth 1 -name '*.csv' | wc -l)"
echo "JUNCTION_REPLAYS=$(find "$REPLAY" -maxdepth 1 -name '*.json' | wc -l)"
