#!/usr/bin/env bash
set -euo pipefail
cd /root/SpaLORA-night16h
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=/root/SpaLORA-night16h
PY=/root/miniconda3/envs/SpaLORA/bin/python
OUT=/root/night21c_working/replays
mkdir -p "$OUT"
for artifact in /root/night21c_working/formal/*.npz; do
  stem="$(basename "$artifact" .npz)"; lane="${stem%%__*}"
  case "$lane" in
    PLACENTA_K10) carrier=/root/night18d_working/carrier/placenta_carrier.npz ;;
    A1_K10) carrier=/root/night18a_working/carriers/A1_K10.npz ;;
    TONSIL_S1_K4) carrier=/root/night21a_working/carriers/TONSIL_S1_K4.npz ;;
    P22_K9) carrier=/root/night18a_working/carriers/P22_K9.npz ;;
    *) echo "unknown lane $lane" >&2; exit 2 ;;
  esac
  "$PY" scripts/night21c/night21c_official_replay.py --artifact "$artifact" --checkpoint "${artifact%.npz}.pt" \
    --carrier "$carrier" --snapshot-root third_party/night21c_spamgcn_fixed --device cuda --output "$OUT/${stem}.replay.json"
done
echo "REPLAYS=$(find "$OUT" -maxdepth 1 -name '*.replay.json' | wc -l)"
