import json
from pathlib import Path

import pandas as pd

from scripts.night7b_adapter_evaluate import METRICS, final_gate, spatial_gate, summarize


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "protocols/night7b/SpaLORA_Night7B_Adaptive_Relational_Fusion_Registry_2026-08-18.json"


def test_registry_uniqueness_and_exact_budgets():
    registry = json.loads(REGISTRY.read_text())
    heads = [x["id"] for x in registry["head_candidates"]]
    recipes = [x["id"] for x in registry["adapter_recipes"]]
    assert heads == ["H%02d" % i for i in range(18)]
    assert recipes == ["R%02d" % i for i in range(10)]
    assert len(set(heads)) == 18 and len(set(recipes)) == 10
    assert 18 * 30 == registry["budgets"]["H_formal_transforms"]
    assert 10 * 8 == registry["budgets"]["R1_scientific_training"]
    assert 10 * 8 * 2 * 2 == registry["budgets"]["R1_partition_transforms_max"]
    assert 4 * 22 == registry["budgets"]["R2_scientific_training_max"]
    assert 4 * 22 == registry["budgets"]["R2_partition_transforms_max"]


def test_final_gate_is_fail_closed_at_registered_thresholds():
    values = {
        "complete_eligible":True, "priority_weighted_delta_q":.018,
        "a1_mean_delta_q":.001, "d1_mean_delta_q":.001,
        "p22_mean_delta_q":.03, "tonsil_mean_delta_q":-.005,
        "total_q_wins":22, "a1_wins_q":3, "tonsil_wins_q":3,
        "d1_wins_q":6, "p22_wins_q":7,
    }
    for dataset in ("a1", "tonsil", "d1", "p22"):
        values[dataset + "_mean_delta_neighbor_agreement"] = -.01
        values[dataset + "_mean_delta_moran_i"] = -.02
        values[dataset + "_mean_delta_geary_c"] = .02
        values[dataset + "_mean_delta_boundary_disagreement"] = .01
    passed, _ = final_gate(pd.Series(values))
    assert passed
    values["priority_weighted_delta_q"] = .017999999
    assert not final_gate(pd.Series(values))[0]


def test_spatial_gate_directions_are_not_reversed():
    values = {}
    for dataset in ("a1", "tonsil", "d1", "p22"):
        values[dataset + "_mean_delta_neighbor_agreement"] = 0
        values[dataset + "_mean_delta_moran_i"] = 0
        values[dataset + "_mean_delta_geary_c"] = 0
        values[dataset + "_mean_delta_boundary_disagreement"] = 0
    assert spatial_gate(pd.Series(values))[0]
    values["p22_mean_delta_geary_c"] = .0200001
    assert not spatial_gate(pd.Series(values))[0]


def test_worker_is_zero_label_and_no_retry_fallback_contract():
    worker = (ROOT / "scripts/night7b_train.py").read_text().lower()
    runner = (ROOT / "scripts/night7b_adapter_stage.py").read_text().lower()
    assert "import anndata" not in worker
    assert "night7b_adapter_evaluate" not in worker
    assert '"label_path"' in worker and '"original_h5ad"' in worker
    compact = runner.replace(" ", "")
    assert '"fallback":false' in compact
    assert '"retry":false' in compact


def test_dataset_first_weighting_does_not_double_ten_seed_datasets():
    counts = {"a1":5, "tonsil":5, "d1":10, "p22":10}
    values = {"a1":.1, "tonsil":.2, "d1":.3, "p22":.4}
    candidate, reference = [], []
    for dataset, count in counts.items():
        for seed in range(count):
            row = {"config_id":"X", "dataset":dataset, "seed":seed, "success":True,
                   "training_runtime_seconds":1.0, "peak_gpu_mib":2.0}
            ref = {"config_id":"H00", "dataset":dataset, "seed":seed}
            for name in METRICS:
                row[name] = values[dataset]
                ref[name] = 0.0
            candidate.append(row); reference.append(ref)
    result = summarize(pd.DataFrame(candidate), pd.DataFrame(reference), ["X"], "H00").iloc[0]
    assert abs(result.priority_weighted_delta_q - .27) < 1e-12
    assert abs(result.balanced_macro_delta_q - .25) < 1e-12
