#!/usr/bin/env bash
set -euo pipefail

repo=${1:-/root/SpaLORA-night16f}
work=${2:-/root/night16f_working}
python=${NIGHT16F_PYTHON:-/root/miniconda3/envs/SpaLORA/bin/python}
export PYTHONPATH="$repo"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
registry="$repo/configs/night16e/chromatin_family_frozen_v2.json"
superseded="$work/superseded/thread_unpinned_formal"

mkdir -p "$superseded"
for name in carriers development formal replays logs; do
  if [ -e "$work/$name" ]; then
    if [ -e "$superseded/$name" ]; then
      echo "refusing to overwrite $superseded/$name" >&2
      exit 2
    fi
    mv "$work/$name" "$superseded/$name"
  fi
done
mkdir -p "$work/carriers" "$work/development" "$work/formal" "$work/replays/run2" "$work/logs"

build_kit() {
  local data_id=$1 lane=$2 k=$3 output=$4 log=$5
  "$python" "$repo/scripts/night16f/build_numeric_carrier.py" kit \
    --kit-root /root/night16d_assets/kit/local_compute_kit \
    --retained-root /root/night16d_assets/retained \
    --starts-root /root/night16d_assets/starts \
    --data-id "$data_id" --lane "$lane" --k "$k" --output "$output" >"$log" 2>&1
}

build_kit P22 P22 9 "$work/carriers/P22_K9.npz" "$work/logs/carrier_p22.log" & p1=$!
build_kit MISAR_E15_5_S1 MISAR_E15_5_S1 7 "$work/carriers/MISAR_K7.npz" "$work/logs/carrier_misar.log" & p2=$!
"$python" "$repo/scripts/night16f/build_numeric_carrier.py" h5ad \
  --rna /root/night16e_external/human_hippocampus/Human_RNA.h5ad \
  --atac /root/night16e_external/human_hippocampus/Human_ATAC_lsi.h5ad \
  --data-id MULTIGATE_HUMAN_HIPPOCAMPUS --lane HUMAN_HIPPOCAMPUS_K7 --k 7 \
  --authority-partition-bank /root/night16e_working/final_replays/human_run1/partitions.npz \
  --authority-partition-index 0 --authority-start-id NIGHT16E_BYTE_EXACT_UNLABELED_START \
  --output "$work/carriers/HUMAN_HIPPOCAMPUS_K7.npz" >"$work/logs/carrier_human.log" 2>&1 & p3=$!
"$python" "$repo/scripts/night16f/build_numeric_carrier.py" h5ad \
  --rna "$work/melanoma_protocol/HumanMelanoma_RNA_sanitized.h5ad" \
  --atac "$work/melanoma_protocol/HumanMelanoma_ATAC_lsi_sanitized.h5ad" \
  --data-id SCP2176_HUMAN_MELANOMA_TUMOR_ONLY --lane MELANOMA_TUMOR_K2 --k 2 \
  --output "$work/carriers/MELANOMA_TUMOR_K2.npz" >"$work/logs/carrier_melanoma.log" 2>&1 & p4=$!
for process in "$p1" "$p2" "$p3" "$p4"; do wait "$process"; done

produce() {
  local carrier=$1 data_id=$2 lane=$3 k=$4 root=$5 log=$6
  mkdir -p "$root/$lane"
  "$python" "$repo/scripts/night16f/night16f_producer.py" \
    --carrier "$carrier" --registry "$registry" --data-id "$data_id" \
    --lane "$lane" --k "$k" --output "$root/$lane/partitions.npz" >"$log" 2>&1
}

produce "$work/carriers/P22_K9.npz" P22 P22_K9 9 "$work/development" "$work/logs/producer_P22_K9.log" & p1=$!
produce "$work/carriers/MISAR_K7.npz" MISAR_E15_5_S1 MISAR_K7 7 "$work/development" "$work/logs/producer_MISAR_K7.log" & p2=$!
produce "$work/carriers/HUMAN_HIPPOCAMPUS_K7.npz" MULTIGATE_HUMAN_HIPPOCAMPUS HUMAN_HIPPOCAMPUS_K7 7 "$work/development" "$work/logs/producer_HUMAN_HIPPOCAMPUS_K7.log" & p3=$!
produce "$work/carriers/MELANOMA_TUMOR_K2.npz" SCP2176_HUMAN_MELANOMA_TUMOR_ONLY MELANOMA_TUMOR_K2 2 "$work/formal" "$work/logs/producer_MELANOMA_TUMOR_K2.log" & p4=$!
for process in "$p1" "$p2" "$p3" "$p4"; do wait "$process"; done

"$python" "$repo/scripts/night16f/night16f_evaluator.py" \
  --carrier "$work/carriers/P22_K9.npz" --partition-bank "$work/development/P22_K9/partitions.npz" \
  --producer-json "$work/development/P22_K9/partitions.producer.json" \
  --reference /root/night16d_assets/kit/local_compute_kit/P22.npz --reference-kind npz \
  --reference-id-key ids --label-key labels_primary --mask-key label_mask \
  --data-id P22 --lane P22_K9 --k 9 --output "$work/development/P22_K9/evaluation.csv"
"$python" "$repo/scripts/night16f/night16f_evaluator.py" \
  --carrier "$work/carriers/MISAR_K7.npz" --partition-bank "$work/development/MISAR_K7/partitions.npz" \
  --producer-json "$work/development/MISAR_K7/partitions.producer.json" \
  --reference /root/night16d_assets/kit/local_compute_kit/MISAR_E15_5_S1.npz --reference-kind npz \
  --reference-id-key ids --label-key labels_primary --mask-key label_mask \
  --data-id MISAR_E15_5_S1 --lane MISAR_K7 --k 7 --output "$work/development/MISAR_K7/evaluation.csv"
"$python" "$repo/scripts/night16f/night16f_evaluator.py" \
  --carrier "$work/carriers/HUMAN_HIPPOCAMPUS_K7.npz" \
  --partition-bank "$work/development/HUMAN_HIPPOCAMPUS_K7/partitions.npz" \
  --producer-json "$work/development/HUMAN_HIPPOCAMPUS_K7/partitions.producer.json" \
  --reference /root/night16e_external/human_hippocampus/human_adata1_official_result.h5ad \
  --reference-kind h5ad --label-key true_label --data-id MULTIGATE_HUMAN_HIPPOCAMPUS \
  --lane HUMAN_HIPPOCAMPUS_K7 --k 7 --output "$work/development/HUMAN_HIPPOCAMPUS_K7/evaluation.csv"
"$python" "$repo/scripts/night16f/night16f_evaluator.py" \
  --carrier "$work/carriers/MELANOMA_TUMOR_K2.npz" \
  --partition-bank "$work/formal/MELANOMA_TUMOR_K2/partitions.npz" \
  --producer-json "$work/formal/MELANOMA_TUMOR_K2/partitions.producer.json" \
  --reference "$work/melanoma_protocol/tumor_k2_reference.tsv" --reference-kind tsv \
  --reference-id-key cell_id --label-key public_author_cluster \
  --data-id SCP2176_HUMAN_MELANOMA_TUMOR_ONLY --lane MELANOMA_TUMOR_K2 --k 2 \
  --output "$work/formal/MELANOMA_TUMOR_K2/evaluation.csv"

for spec in \
  "P22_K9:$work/carriers/P22_K9.npz:P22:9" \
  "MISAR_K7:$work/carriers/MISAR_K7.npz:MISAR_E15_5_S1:7" \
  "HUMAN_HIPPOCAMPUS_K7:$work/carriers/HUMAN_HIPPOCAMPUS_K7.npz:MULTIGATE_HUMAN_HIPPOCAMPUS:7" \
  "MELANOMA_TUMOR_K2:$work/carriers/MELANOMA_TUMOR_K2.npz:SCP2176_HUMAN_MELANOMA_TUMOR_ONLY:2"; do
  IFS=: read -r lane carrier data_id k <<<"$spec"
  produce "$carrier" "$data_id" "$lane" "$k" "$work/replays/run2" "$work/replays/run2/$lane/producer.log"
done

"$python" "$repo/scripts/night16f/audit_exact_replay.py" \
  --pair "P22_K9::$work/development/P22_K9/partitions.npz::$work/replays/run2/P22_K9/partitions.npz" \
  --pair "MISAR_K7::$work/development/MISAR_K7/partitions.npz::$work/replays/run2/MISAR_K7/partitions.npz" \
  --pair "HUMAN_HIPPOCAMPUS_K7::$work/development/HUMAN_HIPPOCAMPUS_K7/partitions.npz::$work/replays/run2/HUMAN_HIPPOCAMPUS_K7/partitions.npz" \
  --pair "MELANOMA_TUMOR_K2::$work/formal/MELANOMA_TUMOR_K2/partitions.npz::$work/replays/run2/MELANOMA_TUMOR_K2/partitions.npz" \
  --output "$work/replays/exact_replay_audit.json"

date -u +%FT%TZ >"$work/development/COMPLETE"
date -u +%FT%TZ >"$work/formal/COMPLETE"
