#!/usr/bin/env bash
set -euo pipefail
cd /root/SpaLORA-night16h
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=/root/SpaLORA-night16h
PY=/root/miniconda3/envs/SpaLORA/bin/python
OUT=/root/night21c_working/endpoint_banks
mkdir -p "$OUT"

lock_one() {
  local lane="$1" k="$2" carrier="$3" embedding="$4" key="$5" source="$6"
  local target="$OUT/${lane}__${source}.npz"
  "$PY" scripts/night21c/night21c_endpoint_producer.py --embedding "$embedding" --embedding-key "$key" \
    --carrier "$carrier" --lane "$lane" --representation-source "$source" --k "$k" --output "$target"
}

lock_lane() {
  local lane="$1" k="$2" carrier="$3" retained_artifact="$4" sparse="$5"
  lock_one "$lane" "$k" "$carrier" "$retained_artifact" representation RETAINED_CARRIER
  lock_one "$lane" "$k" "$carrier" "$sparse" representation NIGHT21B_SPARSE_PORT_STABLE700
  while IFS= read -r artifact; do
    local stem source
    stem="$(basename "$artifact" .npz)"; source="${stem#${lane}__}"
    lock_one "$lane" "$k" "$carrier" "$artifact" representation "$source"
  done < <(find /root/night21c_working/formal -maxdepth 1 -type f -name "${lane}__*.npz" | sort)
}

lock_lane PLACENTA_K10 10 /root/night18d_working/carrier/placenta_carrier.npz /root/night21b_working/discovery/PLACENTA_K10__T0_STRONG_CARRIER__STABLE700_V1_S0.npz /root/night21b_working/discovery/PLACENTA_K10__B0_BACKBONE_ONLY__STABLE700_V1_S0.npz
lock_lane A1_K10 10 /root/night18a_working/carriers/A1_K10.npz /root/night21b_working/discovery/A1_K10__T0_STRONG_CARRIER__STABLE700_V1_S0.npz /root/night21b_working/discovery/A1_K10__B0_BACKBONE_ONLY__STABLE700_V1_S0.npz
lock_lane TONSIL_S1_K4 4 /root/night21a_working/carriers/TONSIL_S1_K4.npz /root/night21b_working/discovery/TONSIL_S1_K4__T0_STRONG_CARRIER__STABLE700_V1_S0.npz /root/night21b_working/discovery/TONSIL_S1_K4__B0_BACKBONE_ONLY__STABLE700_V1_S0.npz
lock_lane P22_K9 9 /root/night18a_working/carriers/P22_K9.npz /root/night21b_working/discovery/P22_K9__T0_STRONG_CARRIER__STABLE700_V1_S0.npz /root/night21b_working/discovery/P22_K9__B0_BACKBONE_ONLY__STABLE700_V1_S0.npz

python_files=$(find "$OUT" -maxdepth 1 -type f -name '*.npz' | wc -l)
echo "LOCKED_ENDPOINT_BANKS=$python_files"
