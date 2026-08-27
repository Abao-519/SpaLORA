#!/usr/bin/env bash
set -euo pipefail
cd /root/SpaLORA-night16h
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0
export PYTHONPATH=/root/SpaLORA-night16h
PY=/root/miniconda3/envs/SpaLORA/bin/python
OUT=/root/night22a_working/geometry
EVAL=/root/night22a_working/evaluations
mkdir -p "$OUT" "$EVAL"

produce() {
  local lane="$1" k="$2" carrier="$3" source="$4" embedding="$5" parent="$6"
  "$PY" scripts/night22a/night22a_geometry_producer.py \
    --lane "$lane" --k "$k" --carrier "$carrier" --representation-source "$source" \
    --embedding "$embedding" --parent-bank "$parent" --output "$OUT/${lane}__${source}.npz"
}

produce A1_K10 10 /root/night18a_working/carriers/A1_K10.npz RETAINED_CARRIER \
  /root/night21b_working/discovery/A1_K10__T0_STRONG_CARRIER__STABLE700_V1_S0.npz \
  /root/night21c_working/endpoint_banks/A1_K10__RETAINED_CARRIER.npz
produce A1_K10 10 /root/night18a_working/carriers/A1_K10.npz NIGHT21B_SPARSE_PORT_STABLE700 \
  /root/night21b_working/discovery/A1_K10__B0_BACKBONE_ONLY__STABLE700_V1_S0.npz \
  /root/night21c_working/endpoint_banks/A1_K10__NIGHT21B_SPARSE_PORT_STABLE700.npz
produce A1_K10 10 /root/night18a_working/carriers/A1_K10.npz OFFICIAL_CONFIG_DEFAULT_700__OFFICIAL_FULL \
  /root/night21c_working/formal/A1_K10__OFFICIAL_CONFIG_DEFAULT_700__OFFICIAL_FULL.npz \
  /root/night21c_working/endpoint_banks/A1_K10__OFFICIAL_CONFIG_DEFAULT_700__OFFICIAL_FULL.npz

produce TONSIL_S1_K4 4 /root/night21a_working/carriers/TONSIL_S1_K4.npz RETAINED_CARRIER \
  /root/night21b_working/discovery/TONSIL_S1_K4__T0_STRONG_CARRIER__STABLE700_V1_S0.npz \
  /root/night21c_working/endpoint_banks/TONSIL_S1_K4__RETAINED_CARRIER.npz
produce TONSIL_S1_K4 4 /root/night21a_working/carriers/TONSIL_S1_K4.npz NIGHT21B_SPARSE_PORT_STABLE700 \
  /root/night21b_working/discovery/TONSIL_S1_K4__B0_BACKBONE_ONLY__STABLE700_V1_S0.npz \
  /root/night21c_working/endpoint_banks/TONSIL_S1_K4__NIGHT21B_SPARSE_PORT_STABLE700.npz
produce TONSIL_S1_K4 4 /root/night21a_working/carriers/TONSIL_S1_K4.npz S1_NOTEBOOK_EXACT_300__OFFICIAL_AE_ONLY \
  /root/night21c_working/formal/TONSIL_S1_K4__S1_NOTEBOOK_EXACT_300__OFFICIAL_AE_ONLY.npz \
  /root/night21c_working/endpoint_banks/TONSIL_S1_K4__S1_NOTEBOOK_EXACT_300__OFFICIAL_AE_ONLY.npz

produce P22_K9 9 /root/night18a_working/carriers/P22_K9.npz RETAINED_CARRIER \
  /root/night21b_working/discovery/P22_K9__T0_STRONG_CARRIER__STABLE700_V1_S0.npz \
  /root/night21c_working/endpoint_banks/P22_K9__RETAINED_CARRIER.npz
produce P22_K9 9 /root/night18a_working/carriers/P22_K9.npz NIGHT21B_SPARSE_PORT_STABLE700 \
  /root/night21b_working/discovery/P22_K9__B0_BACKBONE_ONLY__STABLE700_V1_S0.npz \
  /root/night21c_working/endpoint_banks/P22_K9__NIGHT21B_SPARSE_PORT_STABLE700.npz
produce P22_K9 9 /root/night18a_working/carriers/P22_K9.npz MOUSE_E15_NOTEBOOK_ANALOG_400__OFFICIAL_GRAPH_ONLY \
  /root/night21c_working/formal/P22_K9__MOUSE_E15_NOTEBOOK_ANALOG_400__OFFICIAL_GRAPH_ONLY.npz \
  /root/night21c_working/endpoint_banks/P22_K9__MOUSE_E15_NOTEBOOK_ANALOG_400__OFFICIAL_GRAPH_ONLY.npz
produce P22_K9 9 /root/night18a_working/carriers/P22_K9.npz MOUSE_E15_NOTEBOOK_ANALOG_400__OFFICIAL_FULL \
  /root/night21c_working/formal/P22_K9__MOUSE_E15_NOTEBOOK_ANALOG_400__OFFICIAL_FULL.npz \
  /root/night21c_working/endpoint_banks/P22_K9__MOUSE_E15_NOTEBOOK_ANALOG_400__OFFICIAL_FULL.npz

produce PLACENTA_K10 10 /root/night18d_working/carrier/placenta_carrier.npz RETAINED_CARRIER \
  /root/night21b_working/discovery/PLACENTA_K10__T0_STRONG_CARRIER__STABLE700_V1_S0.npz \
  /root/night21c_working/endpoint_banks/PLACENTA_K10__RETAINED_CARRIER.npz
produce PLACENTA_K10 10 /root/night18d_working/carrier/placenta_carrier.npz NIGHT21B_SPARSE_PORT_STABLE700 \
  /root/night21b_working/discovery/PLACENTA_K10__B0_BACKBONE_ONLY__STABLE700_V1_S0.npz \
  /root/night21c_working/endpoint_banks/PLACENTA_K10__NIGHT21B_SPARSE_PORT_STABLE700.npz
produce PLACENTA_K10 10 /root/night18d_working/carrier/placenta_carrier.npz MOUSE_E15_NOTEBOOK_ANALOG_400__OFFICIAL_AE_ONLY \
  /root/night21c_working/formal/PLACENTA_K10__MOUSE_E15_NOTEBOOK_ANALOG_400__OFFICIAL_AE_ONLY.npz \
  /root/night21c_working/endpoint_banks/PLACENTA_K10__MOUSE_E15_NOTEBOOK_ANALOG_400__OFFICIAL_AE_ONLY.npz

echo "LOCKED_GEOMETRY_BANKS=$(find "$OUT" -maxdepth 1 -name '*.npz' | wc -l)"

evaluate() {
  local lane="$1" authority="$2" reference="${3:-}"
  while IFS= read -r bank; do
    local stem args
    stem="$(basename "$bank" .npz)"
    args=(--bank "$bank" --authority "$authority" --output "$EVAL/${stem}.csv")
    if [[ -n "$reference" ]]; then args+=(--reference-h5ad "$reference"); fi
    "$PY" scripts/night22a/night22a_geometry_evaluator.py "${args[@]}"
  done < <(find "$OUT" -maxdepth 1 -name "${lane}__*.npz" | sort)
}

evaluate A1_K10 /root/night16d_assets/kit/local_compute_kit/A1.npz
evaluate TONSIL_S1_K4 /root/night16d_assets/kit/local_compute_kit/tonsil_s1.npz
evaluate P22_K9 /root/night16d_assets/kit/local_compute_kit/P22.npz
evaluate PLACENTA_K10 /root/night18d_working/carrier/placenta_carrier.npz "/root/autodl-fs/Human placenta architecture/humanplacenta_rna.h5ad"
echo "GEOMETRY_EVALUATIONS=$(find "$EVAL" -maxdepth 1 -name '*.csv' | wc -l)"
