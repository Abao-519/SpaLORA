#!/usr/bin/env bash
set -euo pipefail
cd /root/SpaLORA-night16h
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=/root/SpaLORA-night16h
PY=/root/miniconda3/envs/SpaLORA/bin/python
SNAP=third_party/night21c_spamgcn_fixed
OUT=/root/night21c_working/formal
mkdir -p "$OUT"

run_one() {
  local lane="$1" k="$2" carrier="$3" profile="$4" arm="$5"
  local profile_id
  profile_id="$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["config"]["profile_id"])' "$profile")"
  local target="$OUT/${lane}__${profile_id}__${arm}.npz"
  if [[ -s "$target" && -s "${target%.npz}.pt" && -s "${target%.npz}.json" ]]; then
    echo "SKIP_COMPLETE $target"
    return
  fi
  echo "START $target $(date -u +%FT%TZ)"
  "$PY" scripts/night21c/night21c_official_producer.py \
    --carrier "$carrier" --lane "$lane" --k "$k" --profile "$profile" --arm "$arm" \
    --snapshot-root "$SNAP" --training-seed 0 --endpoint-seed 0 --device cuda --output "$target"
  echo "DONE $target $(date -u +%FT%TZ)"
}

for arm in OFFICIAL_FULL OFFICIAL_AE_ONLY OFFICIAL_GRAPH_ONLY; do
  run_one PLACENTA_K10 10 /root/night18d_working/carrier/placenta_carrier.npz configs/night21c/profiles/placenta_notebook.json "$arm"
done

run_one A1_K10 10 /root/night18a_working/carriers/A1_K10.npz configs/night21c/profiles/default700.json OFFICIAL_FULL
for arm in OFFICIAL_FULL OFFICIAL_AE_ONLY OFFICIAL_GRAPH_ONLY; do
  run_one A1_K10 10 /root/night18a_working/carriers/A1_K10.npz configs/night21c/profiles/a1_notebook.json "$arm"
done

run_one TONSIL_S1_K4 4 /root/night21a_working/carriers/TONSIL_S1_K4.npz configs/night21c/profiles/default700.json OFFICIAL_FULL
for arm in OFFICIAL_FULL OFFICIAL_AE_ONLY OFFICIAL_GRAPH_ONLY; do
  run_one TONSIL_S1_K4 4 /root/night21a_working/carriers/TONSIL_S1_K4.npz configs/night21c/profiles/tonsil_s1_notebook.json "$arm"
done

run_one P22_K9 9 /root/night18a_working/carriers/P22_K9.npz configs/night21c/profiles/default700.json OFFICIAL_FULL
for arm in OFFICIAL_FULL OFFICIAL_AE_ONLY OFFICIAL_GRAPH_ONLY; do
  run_one P22_K9 9 /root/night18a_working/carriers/P22_K9.npz configs/night21c/profiles/p22_notebook.json "$arm"
done
