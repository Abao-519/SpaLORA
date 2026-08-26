#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=/root/SpaLORA-night16h
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

repo=/root/SpaLORA-night16h
work=/root/night19a_working/d0_rev1
python=/root/miniconda3/envs/SpaLORA/bin/python
contract="$repo/configs/night19a/d0_formula_freeze_rev1.json"
mkdir -p "$work"
cd "$repo"

run_lane() {
  local lane="$1" carrier="$2" bank="$3" feasibility="$4" source="$5" k="$6" seed="$7"
  local out="$work/$lane/S$seed"
  mkdir -p "$out"
  local feasibility_args=()
  if [[ -n "$feasibility" ]]; then
    feasibility_args=(--feasibility "$feasibility")
  fi
  "$python" scripts/night19a/night19a_d0_producer.py \
    --lane "$lane" --carrier "$carrier" --candidate-bank "$bank" \
    "${feasibility_args[@]}" --relation-source "$source" --k "$k" \
    --training-seed "$seed" --execution-contract "$contract" \
    --output-dir "$out" --device cuda >"$out/producer.log" 2>&1
  "$python" scripts/night19a/night19a_d0_replay.py \
    --carrier "$carrier" --candidate-bank "$bank" "${feasibility_args[@]}" \
    --artifact "$out/artifact.npz" --checkpoint "$out/checkpoint.pt" \
    --producer-json "$out/producer.json" --output "$out/fresh_replay.json" \
    --device cuda >"$out/replay.log" 2>&1
  "$python" - "$out/producer.json" "$out/fresh_replay.json" <<'PY'
import json, sys
p=json.load(open(sys.argv[1], encoding='utf-8'))
r=json.load(open(sys.argv[2], encoding='utf-8'))
assert p['execution_contract_sha256'] == '8d15260779f20ef322b70eacc313ad02d2e90241c5f180f3f8881f366955b1e1'
assert r['partition_exact'] and r['representation_exact'] and r['ordered_ids_exact']
print(p['lane'], p['training_seed'], p['partition_sha256'], 'REV1_REPLAY_PASS')
PY
}

for seed in 0 1; do
  run_lane P22_K9 /root/night16f_working/carriers/P22_K9.npz \
    /root/night16g_working/candidate_features/P22_K9.npz \
    /root/night16h_working/feasibility/P22_K9.csv NIGHT16H_UNBIASED_WEIGHTED 9 "$seed"
  run_lane MISAR_K7 /root/night16f_working/carriers/MISAR_K7.npz \
    /root/night16g_working/candidate_features/MISAR_K7.npz \
    /root/night16h_working/feasibility/MISAR_K7.csv NIGHT16H_UNBIASED_WEIGHTED 7 "$seed"
  run_lane HUMAN_HIPPOCAMPUS_K7 /root/night16f_working/carriers/HUMAN_HIPPOCAMPUS_K7.npz \
    /root/night16g_working/candidate_features/HUMAN_HIPPOCAMPUS_K7.npz \
    /root/night16h_working/feasibility/HUMAN_HIPPOCAMPUS_K7.csv NIGHT16H_UNBIASED_WEIGHTED 7 "$seed"
  run_lane PLACENTA_K10 /root/night18d_working/carrier/placenta_carrier.npz \
    /root/night18e_working/development_formal_v2/PLACENTA_K10/partitions.npz \
    '' UNIFORM_FEASIBLE_STRESS 10 "$seed"
done

manifest_args=()
for lane in P22_K9 MISAR_K7 HUMAN_HIPPOCAMPUS_K7 PLACENTA_K10; do
  for seed in 0 1; do
    manifest_args+=(--manifest "$work/$lane/S$seed/producer.json")
  done
done
"$python" scripts/night19a/build_d0_gate_rev1.py \
  "${manifest_args[@]}" --contract "$contract" --output "$work/d0_gate_rev1.json" \
  >"$work/d0_gate_rev1.log" 2>&1
"$python" - "$work/d0_gate_rev1.json" <<'PY'
import json, sys
x=json.load(open(sys.argv[1], encoding='utf-8'))
print(json.dumps({
  'schema': x['schema'],
  'primary_lane_pass_count': x['primary_lane_pass_count'],
  'stage_a_authorized': x['stage_a_authorized'],
  'lane_pass': {r['lane']: r['lane_pass'] for r in x['lane_results']},
  'reproducible_pairs': {r['lane']: r['reproducible_critical_pairs'] for r in x['lane_results']},
}, indent=2, sort_keys=True))
PY
du -sh "$work"
df -h /
