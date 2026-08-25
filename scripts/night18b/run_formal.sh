#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=/root/SpaLORA-night16h
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0
repo=/root/SpaLORA-night16h
work=/root/night18b_working/formal
mkdir -p "$work/artifacts" "$work/evaluation" "$work/replay" "$work/logs"
cd "$repo"
arms=(MATCHED_BACKBONE FULL_RESPONSE_CALIBRATION LOWPASS_ONLY_CALIBRATION SHARPEN_ONLY_CALIBRATION SHARED_SCALE_CONTROL SWAPPED_RESPONSE_CONTROL)
for lane_k in A1_K10:10 P22_K9:9; do
  lane=${lane_k%%:*}; k=${lane_k##*:}
  carrier="/root/night18a_working/carriers/${lane}.npz"
  authority_lane=${lane%%_K*}
  authority="/root/night16d_assets/kit/local_compute_kit/${authority_lane}.npz"
  for arm in "${arms[@]}"; do
    stem="${lane}__${arm}__S0"
    if [[ "$stem" == "A1_K10__FULL_RESPONSE_CALIBRATION__S0" ]]; then
      cp /root/night18b_working/p0_A1/FULL_RESPONSE_CALIBRATION.npz "$work/artifacts/${stem}.npz"
      cp /root/night18b_working/p0_A1/FULL_RESPONSE_CALIBRATION.pt "$work/artifacts/${stem}.pt"
      cp /root/night18b_working/p0_A1/FULL_RESPONSE_CALIBRATION.json "$work/artifacts/${stem}.json"
    else
      /root/miniconda3/envs/SpaLORA/bin/python scripts/night18b/night18b_transfer_producer.py \
        --carrier "$carrier" --lane "$lane" --k "$k" --arm "$arm" --seed 0 \
        --output "$work/artifacts/${stem}.npz" >"$work/logs/${stem}.producer.log" 2>&1
    fi
    /root/miniconda3/envs/SpaLORA/bin/python scripts/night18b/night18b_transfer_replay.py \
      --carrier "$carrier" --artifact "$work/artifacts/${stem}.npz" \
      --manifest "$work/artifacts/${stem}.json" --checkpoint "$work/artifacts/${stem}.pt" \
      --k "$k" --output "$work/replay/${stem}.json" >"$work/logs/${stem}.replay.log" 2>&1
    /root/miniconda3/envs/SpaLORA/bin/python scripts/night18b/night18b_transfer_evaluator.py \
      --artifact "$work/artifacts/${stem}.npz" --manifest "$work/artifacts/${stem}.json" \
      --authority "$authority" --k "$k" --output "$work/evaluation/${stem}.csv" \
      >"$work/logs/${stem}.evaluator.log" 2>&1
  done
done
touch "$work/FORMAL_COMPLETE"
