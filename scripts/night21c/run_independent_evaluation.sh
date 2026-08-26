#!/usr/bin/env bash
set -euo pipefail
cd /root/SpaLORA-night16h
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=/root/SpaLORA-night16h
PY=/root/miniconda3/envs/SpaLORA/bin/python
BANK=/root/night21c_working/endpoint_banks
EVAL=/root/night21c_working/evaluations
PROBE=/root/night21c_working/probes
mkdir -p "$EVAL" "$PROBE"

evaluate_lane() {
  local lane="$1" authority="$2" reference="${3:-}"
  while IFS= read -r bank; do
    local stem source embedding key args
    stem="$(basename "$bank" .npz)"; source="${stem#${lane}__}"
    args=(--bank "$bank" --authority "$authority" --output "$EVAL/${stem}.csv")
    if [[ -n "$reference" ]]; then args+=(--reference-h5ad "$reference"); fi
    "$PY" scripts/night21c/night21c_endpoint_evaluator.py "${args[@]}"
    embedding="$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["embedding_path"])' "${bank%.npz}.json")"
    key=representation
    args=(--embedding "$embedding" --embedding-key "$key" --authority "$authority" --lane "$lane" --representation-source "$source" --output "$PROBE/${stem}.csv")
    if [[ -n "$reference" ]]; then args+=(--reference-h5ad "$reference"); fi
    "$PY" scripts/night21c/night21c_label_after_lock_probe.py "${args[@]}"
  done < <(find "$BANK" -maxdepth 1 -type f -name "${lane}__*.npz" | sort)
}

evaluate_lane A1_K10 /root/night16d_assets/kit/local_compute_kit/A1.npz
evaluate_lane TONSIL_S1_K4 /root/night16d_assets/kit/local_compute_kit/tonsil_s1.npz
evaluate_lane P22_K9 /root/night16d_assets/kit/local_compute_kit/P22.npz
evaluate_lane PLACENTA_K10 /root/night18d_working/carrier/placenta_carrier.npz "/root/autodl-fs/Human placenta architecture/humanplacenta_rna.h5ad"
echo "EVALUATIONS=$(find "$EVAL" -maxdepth 1 -name '*.csv' | wc -l) PROBES=$(find "$PROBE" -maxdepth 1 -name '*.csv' | wc -l)"
