#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=/root/SpaLORA-night16h
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

repo=/root/SpaLORA-night16h
work=/root/night19a_working/stage_a_formal
python=/root/miniconda3/envs/SpaLORA/bin/python
contract="$repo/configs/night19a/stage_a_formula_freeze_rev1.json"
d0_gate=/root/night19a_working/d0_rev1/d0_gate_rev1.json
mkdir -p "$work"
cd "$repo"

produce() {
  local lane="$1" carrier="$2" bank="$3" feasibility="$4" relation_source="$5" k="$6"
  local out="$work/$lane/S0"
  mkdir -p "$out"
  local feasibility_args=()
  if [[ -n "$feasibility" ]]; then feasibility_args=(--feasibility "$feasibility"); fi
  if [[ -f "$out/fresh_replay.json" ]]; then
    "$python" - "$out/producer.json" "$out/fresh_replay.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding='utf-8')); r=json.load(open(sys.argv[2],encoding='utf-8'))
assert p['execution_contract_sha256']=='d1a5275f8b76865de555d37a939b9e77dcc080590a8fc4bedcb1fd047c8a01a5'
assert r['ids_exact'] and r['profiles_exact'] and r['representations_exact'] and r['partitions_exact']
print(p['lane'], p['artifact_sha256'], 'PREVIOUSLY_COMPLETED_LOCK_AND_REPLAY_REUSED')
PY
    return
  fi
  "$python" scripts/night19a/night19a_stage_a_producer.py \
    --lane "$lane" --carrier "$carrier" --candidate-bank "$bank" "${feasibility_args[@]}" \
    --relation-source "$relation_source" --k "$k" --training-seed 0 \
    --d0-gate "$d0_gate" --execution-contract "$contract" --output-dir "$out" --device cuda \
    >"$out/producer.log" 2>&1
  "$python" scripts/night19a/night19a_stage_a_replay.py \
    --lane "$lane" --carrier "$carrier" --candidate-bank "$bank" "${feasibility_args[@]}" \
    --relation-source "$relation_source" --k "$k" --training-seed 0 \
    --artifact "$out/artifact.npz" --checkpoint "$out/checkpoints.pt" \
    --producer-json "$out/producer.json" --output "$out/fresh_replay.json" --device cuda \
    >"$out/replay.log" 2>&1
  "$python" - "$out/producer.json" "$out/fresh_replay.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding='utf-8')); r=json.load(open(sys.argv[2],encoding='utf-8'))
assert p['execution_contract_sha256']=='d1a5275f8b76865de555d37a939b9e77dcc080590a8fc4bedcb1fd047c8a01a5'
assert r['ids_exact'] and r['profiles_exact'] and r['representations_exact'] and r['partitions_exact']
assert p['permuted_projection_groups']['maximum_absolute_mass_error'] <= 1e-8
assert max(v.get('maximum_relation_gradient_norm_match_error',0) or 0 for v in p['arm_diagnostics'].values()) <= 1e-6
print(p['lane'], p['artifact_sha256'], 'STAGE_A_LOCK_AND_REPLAY_PASS')
PY
}

produce P22_K9 /root/night16f_working/carriers/P22_K9.npz \
  /root/night16g_working/candidate_features/P22_K9.npz /root/night16h_working/feasibility/P22_K9.csv NIGHT16H_UNBIASED_WEIGHTED 9
produce MISAR_K7 /root/night16f_working/carriers/MISAR_K7.npz \
  /root/night16g_working/candidate_features/MISAR_K7.npz /root/night16h_working/feasibility/MISAR_K7.csv NIGHT16H_UNBIASED_WEIGHTED 7
produce HUMAN_HIPPOCAMPUS_K7 /root/night16f_working/carriers/HUMAN_HIPPOCAMPUS_K7.npz \
  /root/night16g_working/candidate_features/HUMAN_HIPPOCAMPUS_K7.npz /root/night16h_working/feasibility/HUMAN_HIPPOCAMPUS_K7.csv NIGHT16H_UNBIASED_WEIGHTED 7
produce PLACENTA_K10 /root/night18d_working/carrier/placenta_carrier.npz \
  /root/night18e_working/development_formal_v2/PLACENTA_K10/partitions.npz '' UNIFORM_FEASIBLE_STRESS 10

# Annotation access starts only after all four producer artifacts and replays are locked.
"$python" scripts/night19a/night19a_stage_a_evaluator.py \
  --artifact "$work/P22_K9/S0/artifact.npz" --producer-json "$work/P22_K9/S0/producer.json" \
  --carrier /root/night16f_working/carriers/P22_K9.npz \
  --reference /root/night16d_assets/kit/local_compute_kit/P22.npz --reference-kind npz \
  --label-key labels_primary --mask-key label_mask --output "$work/P22_K9/S0/evaluation.csv" \
  >"$work/P22_K9/S0/evaluator.log" 2>&1
"$python" scripts/night19a/night19a_stage_a_evaluator.py \
  --artifact "$work/MISAR_K7/S0/artifact.npz" --producer-json "$work/MISAR_K7/S0/producer.json" \
  --carrier /root/night16f_working/carriers/MISAR_K7.npz \
  --reference /root/night16d_assets/kit/local_compute_kit/MISAR_E15_5_S1.npz --reference-kind npz \
  --label-key labels_primary --mask-key label_mask --output "$work/MISAR_K7/S0/evaluation.csv" \
  >"$work/MISAR_K7/S0/evaluator.log" 2>&1
"$python" scripts/night19a/night19a_stage_a_evaluator.py \
  --artifact "$work/HUMAN_HIPPOCAMPUS_K7/S0/artifact.npz" --producer-json "$work/HUMAN_HIPPOCAMPUS_K7/S0/producer.json" \
  --carrier /root/night16f_working/carriers/HUMAN_HIPPOCAMPUS_K7.npz \
  --reference /root/night16e_external/human_hippocampus/human_adata1_official_result.h5ad --reference-kind h5ad \
  --label-key true_label --output "$work/HUMAN_HIPPOCAMPUS_K7/S0/evaluation.csv" \
  >"$work/HUMAN_HIPPOCAMPUS_K7/S0/evaluator.log" 2>&1
"$python" scripts/night19a/night19a_stage_a_evaluator.py \
  --artifact "$work/PLACENTA_K10/S0/artifact.npz" --producer-json "$work/PLACENTA_K10/S0/producer.json" \
  --carrier /root/night18d_working/carrier/placenta_carrier.npz \
  --reference '/autodl-fs/data/Human placenta architecture/humanplacenta_rna.h5ad' --reference-kind h5ad \
  --label-key cell_type --output "$work/PLACENTA_K10/S0/evaluation.csv" \
  >"$work/PLACENTA_K10/S0/evaluator.log" 2>&1

"$python" scripts/night19a/build_stage_a_summary.py \
  --evaluation "$work/P22_K9/S0/evaluation.csv" --evaluation "$work/MISAR_K7/S0/evaluation.csv" \
  --evaluation "$work/HUMAN_HIPPOCAMPUS_K7/S0/evaluation.csv" --evaluation "$work/PLACENTA_K10/S0/evaluation.csv" \
  --output "$work/stage_a_gate.json" >"$work/stage_a_gate.log" 2>&1
"$python" - "$work/stage_a_gate.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
print(json.dumps(x,indent=2,sort_keys=True))
PY
du -sh "$work"
df -h /
