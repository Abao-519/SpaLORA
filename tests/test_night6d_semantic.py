from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from SpaLORA.night3a_ige import model_state_sha256
from SpaLORA.night3af_cache import sha256_file
from SpaLORA.night5a_rnd import Night5ATrainer
from SpaLORA.night6c_pipeline import (
    BASE_C04, _neighbors, binary_knn, forward_model, row_normalize,
    run_head, self_tuning_affinity,
)
from SpaLORA.night6d_firewall import FirewallViolation, guard_path, reject_transform_payload
from SpaLORA.night6d_pipeline import (
    DATASET_CFG, EXPECTED_GRAPH_SHA, EXPECTED_HEAD_SHA, GRAPHS, HEADS,
    canonical_json_sha, validate_locked_contracts,
)

REPO = Path(__file__).resolve().parents[1]


def test_exact_locked_factorial_and_hashes():
    validate_locked_contracts()
    assert list(GRAPHS) == ["G00_SP18_F20_CORR_UNION", "G04_SP10_F10_EUC_UNION"]
    assert list(HEADS) == ["H00_FUSED_PCA20_MCLUST_EEE", "H05_EQUAL3_AFFINITY_SPECTRAL"]
    assert {k: canonical_json_sha(v) for k, v in GRAPHS.items()} == EXPECTED_GRAPH_SHA
    assert {k: canonical_json_sha(v) for k, v in HEADS.items()} == EXPECTED_HEAD_SHA


def test_locked_night6c_implementation_sources():
    expected = {
        "SpaLORA/night6c_pipeline.py": "b1abd06ab1f5f30d8f3c1d16e89a2ab8c240b61422f8c2b9307adc0619a0e0d1",
        "scripts/night6c_train.py": "3603b1d857750ff15f9c39b570f19171b7d59b2f53efbcab04aeb10eb820d2f9",
        "scripts/night6c_reload.py": "32522425a5a56130a67e0b35928f48908e25abdde3b43982dd475a507d3237d9",
        "scripts/night6c_transform.py": "a33494a19bbfb22768a13e87d0111e3ca663bd970c0268aecbeeadff258b5b0e",
    }
    assert {name: sha256_file(REPO / name) for name in expected} == expected


def test_dataset_contracts_are_fixed():
    assert DATASET_CFG["d1"] == {
        "n_clusters": 10, "embedding_dim": 64, "epochs": 200,
        "loss_factors": [1.9, 2.5, 1.5, 10.0], "locked_m_bad_expected": 2.289938091,
    }
    assert DATASET_CFG["p22"] == {
        "n_clusters": 9, "embedding_dim": 128, "epochs": 1600,
        "loss_factors": [1.5, 5.0, 1.5, 1.0], "locked_m_bad_expected": 2.290322960,
    }


def test_knn_lexical_tie_and_locked_graph_rules():
    x = np.array([[0., 0.], [1., 0.], [-1., 0.], [5., 0.], [6., 0.]])
    ids = np.array(["m", "z", "a", "q", "p"])
    assert _neighbors(x, 1, "euclidean", ids)[0, 0] == 2
    union = binary_knn(x, 1, "euclidean", ids, "union")
    assert (union != union.T).nnz == 0 and np.all(union.diagonal() == 0)
    with pytest.raises(ValueError):
        _neighbors(x, 1, "cosine", ids)


def test_self_tuning_kernel_matches_manual_sparse_construction():
    rng = np.random.default_rng(20260817)
    values = rng.normal(size=(32, 7))
    ids = np.array([f"id{i:03d}" for i in range(32)])
    observed = self_tuning_affinity(values, 10, ids)
    x = row_normalize(values)
    idx = _neighbors(x, 10, "euclidean", ids)
    rows = np.repeat(np.arange(len(x)), 10)
    cols = idx.reshape(-1)
    distances = np.linalg.norm(x[rows] - x[cols], axis=1).reshape(len(x), 10)
    sigma = np.maximum(distances[:, -1], 1e-12)
    weight = np.exp(-(distances.reshape(-1) ** 2) /
                    np.maximum(sigma[rows] * sigma[cols], 1e-12))
    manual = sp.coo_matrix((weight, (rows, cols)), shape=(len(x), len(x))).tocsr()
    manual = manual.maximum(manual.T); manual.setdiag(0); manual.eliminate_zeros()
    assert np.allclose(observed.toarray(), manual.toarray(), atol=1e-15, rtol=0)


def test_h05_exact_three_view_arithmetic_mean_and_no_fallback(tmp_path):
    rng = np.random.default_rng(2020)
    true = np.repeat(np.arange(3), 30)
    centers = np.eye(3, 6) * 3.0
    base = centers[true] + rng.normal(scale=.2, size=(90, 6))
    views = {
        "emb_latent_omics1": base + rng.normal(scale=.05, size=base.shape),
        "emb_latent_omics2": base + rng.normal(scale=.05, size=base.shape),
        "SpaLORA_fused": base + rng.normal(scale=.03, size=base.shape),
    }
    coords = np.column_stack((np.arange(90) % 15, np.arange(90) // 15))
    ids = np.array([f"spot{i:03d}" for i in range(90)])
    labels, audit = run_head(HEADS["H05_EQUAL3_AFFINITY_SPECTRAL"], views, 3,
                             coords, ids, tmp_path)
    assert len(np.unique(labels)) == 3
    assert audit["fallback"] is False
    assert (tmp_path / "affinity.npz").is_file()


def test_firewall_rejects_originals_labels_early_evaluator_and_metrics():
    safe = "/root/autodl-fs/night6d_data_20260817/d1_label_free/d1_rna_label_free.h5ad"
    assert guard_path(safe, role="trainer_transformer", operation="read")
    forbidden = [
        "/root/autodl-fs/Human lymph node/D1/humanlymphnode_rna.h5ad",
        "/root/autodl-fs/P22 mouse brain coronal section/mousebrain_rna.h5ad",
        "/root/autodl-fs/Human lymph node/D1/D1_groundtruth.csv",
        "/root/autodl-fs/GSE198353/data.h5ad",
        "/root/autodl-fs/night4b/results.json",
        "/root/autodl-fs/night5d/per_seed_metrics.csv",
    ]
    for path in forbidden:
        with pytest.raises(FirewallViolation):
            guard_path(path, role="trainer_transformer", operation="read")
    with pytest.raises(FirewallViolation):
        guard_path("/root/autodl-fs/Human lymph node/D1/D1_groundtruth.csv",
                   role="evaluator", operation="parse_ground_truth", phase_locked=False)
    for key in ("labels", "single_seed_metric", "intermediate_epoch_metric", "checkpoint_metric"):
        with pytest.raises(FirewallViolation):
            reject_transform_payload(**{key: [1]})


def test_two_step_toy_checkpoint_fresh_process(tmp_path):
    rng = np.random.default_rng(4)
    n = 12
    idx = torch.arange(n, dtype=torch.long)
    identity = torch.sparse_coo_tensor(torch.stack((idx, idx)), torch.ones(n), (n, n)).coalesce()
    data = {
        "features_omics1": rng.normal(size=(n, 5)).astype(np.float32),
        "features_omics2": rng.normal(size=(n, 4)).astype(np.float32),
        "adj_spatial_omics1": identity, "adj_feature_omics1": identity,
        "adj_spatial_omics2": identity, "adj_feature_omics2": identity,
    }
    cfg = {"embedding_dim": 3, "epochs": 2, "loss_factors": [1.9, 2.5, 1.5, 10.0],
           "locked_m_bad_expected": 2.289938091}
    trainer = Night5ATrainer(data, cfg, BASE_C04, 3, torch.device("cpu"), {}, 1e-12)
    result = trainer.train()
    expected = forward_model(result.model, data, torch.device("cpu"))
    checkpoint = tmp_path / "toy.pt"
    torch.save({"state": result.model.state_dict(), "state_sha": model_state_sha256(result.model),
                "data": data, "cfg": cfg, "seed": 3}, checkpoint)
    output = tmp_path / "fresh.npz"
    code = r'''
import numpy as np,sys,torch
from SpaLORA.night5a_rnd import Night5ATrainer
from SpaLORA.night6c_pipeline import BASE_C04,forward_model
from SpaLORA.night3a_ige import model_state_sha256
p=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
t=Night5ATrainer(p['data'],p['cfg'],BASE_C04,p['seed'],torch.device('cpu'),{},1e-12)
m=t.new_model();m.load_state_dict(p['state'],strict=True)
assert model_state_sha256(m)==p['state_sha']
np.savez_compressed(sys.argv[2],**forward_model(m,p['data'],torch.device('cpu')))
'''
    subprocess.run([sys.executable, "-c", code, str(checkpoint), str(output)], cwd=REPO, check=True)
    with np.load(output) as fresh:
        for key, value in expected.items():
            assert np.allclose(value, fresh[key], atol=1e-6, rtol=1e-5)

