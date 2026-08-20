import ast
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from SpaLORA.night6c_pipeline import array_sha
from SpaLORA.night9a_efficient import (
    HEAD_ID, VIEW_KEYS, canonical_operator, eigen_kmeans100, head_config,
    projection_views,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "protocols/night9a/SpaLORA_Night9A_Efficient_Topology_Transfer_Registry_2026-08-20.json"
TASKBOOK = ROOT / "protocols/night9a/SpaLORA_Night9A_Efficient_Topology_Transfer_Taskbook_2026-08-20.md"


def registry():
    return json.loads(REGISTRY.read_text())


def test_authority_and_budget_contracts():
    value = registry()
    assert value["base"]["commit"] == "991ba9dcbd7108c2b3b4c9b4b8e233c5f394da5a"
    assert value["base"]["tag"] == "night8b-cardinality-safe-eval-final-20260820"
    assert value["scientific_budget"] == {
        "formal_chain_units_max": 76,
        "scientific_retry": 0,
        "fallback": 0,
        "global_infrastructure_corrections_max": 4,
        "total_attempts_max": 80,
        "unit_wall_minutes_max": 20,
        "total_wall_hours_max": 6,
    }
    assert value["stages"]["R1"]["units"] == 54
    assert value["stages"]["R2"]["units_max"] == 12
    assert value["stages"]["R3"]["units_max"] == 10
    assert TASKBOOK.is_file()


def test_candidate_registry_exact_and_fixed_epochs():
    candidates = registry()["candidates"]
    assert [x["id"] for x in candidates] == [
        "E00_ZERO_SHOT_TOPOLOGY_SWAP", "E01_WARM_FULL_E40",
        "E02_WARM_FULL_E80", "E03_WARM_FULL_E160", "E04_WARM_FULL_E320",
        "E05_SGC1_RESIDUAL50", "E06_SGC2_MEAN",
        "E07_DELTA_TOPOLOGY_RESIDUAL50", "E08_DELTA_TOPOLOGY_RESIDUAL100",
    ]
    assert [x["target_epochs"] for x in candidates] == [0, 40, 80, 160, 320, 0, 0, 0, 0]
    assert registry()["fixed_backbone_and_adapter"]["no_best_epoch"] is True
    assert registry()["fixed_backbone_and_adapter"]["adapter_recipe"] == "R02_RECON_MNN_EQUAL"
    assert registry()["fixed_backbone_and_adapter"]["adapter_epochs"] == 160


def test_projection_formulas_and_determinism():
    n = 8
    base = sp.diags(np.ones(n - 1), 1) + sp.diags(np.ones(n - 1), -1)
    p00 = canonical_operator(base + sp.diags(np.ones(n - 2), 2) + sp.diags(np.ones(n - 2), -2))
    p04 = canonical_operator(base)
    source = {key: np.arange(n * 4, dtype=np.float32).reshape(n, 4) + index + 1
              for index, key in enumerate(VIEW_KEYS)}
    target = {key: p00 for key in VIEW_KEYS}; old = {key: p04 for key in VIEW_KEYS}
    for candidate in [x["id"] for x in registry()["candidates"][5:]]:
        first = projection_views(candidate, source, target, old)
        second = projection_views(candidate, source, target, old)
        assert set(first) == set(VIEW_KEYS)
        assert {k: array_sha(v) for k, v in first.items()} == {k: array_sha(v) for k, v in second.items()}
        assert all(v.dtype == np.float32 and np.isfinite(v).all() for v in first.values())
        assert all(np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-6) for v in first.values())


def test_same_head_contract_has_only_dataset_locked_k():
    c9 = head_config(9); c12 = head_config(12)
    assert c9["head_id"] == c12["head_id"] == HEAD_ID
    left = json.loads(json.dumps(c9)); right = json.loads(json.dumps(c12))
    for value in (left, right):
        value["eigensolver"].pop("k"); value["kmeans"].pop("n_clusters"); value.pop("dataset_locked_K")
    assert left == right
    affinity = canonical_operator(sp.diags(np.ones(39), 1) + sp.diags(np.ones(39), -1) +
                                  sp.diags(np.ones(38), 2) + sp.diags(np.ones(38), -2))
    first, a = eigen_kmeans100(affinity, 4); second, b = eigen_kmeans100(affinity, 4)
    assert np.array_equal(first, second)
    assert a["canonical_partition_sha256"] == b["canonical_partition_sha256"]


def test_label_firewall_static_contract():
    run_text = (ROOT / "scripts/night9a_run.py").read_text()
    evaluator_text = (ROOT / "scripts/night9a_evaluate.py").read_text()
    module_text = (ROOT / "SpaLORA/night9a_efficient.py").read_text()
    # Pre-label code contains neither MISAR annotation carrier nor an HDF5 reader.
    assert "annotation_carrier" not in run_text + module_text
    assert "h5py" not in run_text + module_text
    assert "anndata" not in run_text + module_text
    # The evaluator is P22-only and cannot address MISAR raw Y.
    assert "p22_labels_locked.npz" in evaluator_text
    assert "annotation_carrier" not in evaluator_text
    assert "h5py" not in evaluator_text
    assert "anndata" not in evaluator_text
    ast.parse(run_text); ast.parse(evaluator_text); ast.parse(module_text)


def test_resource_and_selection_gates_are_locked():
    value = registry()
    gates = value["r1_r2_selection_gates"]
    assert gates["runtime_ratio_vs_u00_max"] == 1.5
    assert gates["peak_gpu_ratio_vs_u00_max"] == 1.25
    assert gates["p22_mean_q_delta_vs_u00_min"] == 0.04
    assert gates["misar_mean_partition_ari_vs_full_f00_min"] == 0.9
    assert value["execution"]["shutdown_last_remote_command"] == "/usr/bin/shutdown"
    assert value["execution"]["autodl_power_api_forbidden"] is True
