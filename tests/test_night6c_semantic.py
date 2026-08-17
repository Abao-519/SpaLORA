from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from SpaLORA.night3a_ige import model_state_sha256
from SpaLORA.night5a_rnd import Night5ATrainer
from SpaLORA.night6c_firewall import FirewallViolation, guard_path, reject_transform_payload
from SpaLORA.night6c_pipeline import (
    BASE_C04, _neighbors, binary_knn, finite_float_or_none, forward_model, moran_scores, normalize_support,
    parse_registry, reliability_weights, run_head, self_tuning_affinity,
)

REPO = Path(__file__).resolve().parents[1]
REG = json.loads((REPO / "protocols/night6c/SpaLORA_Night6B_Candidate_Registry_2026-08-17.json").read_text())


def test_registry_exact_and_unique():
    graphs, heads = parse_registry(REG)
    assert len(graphs) == 9 and len(heads) == 12
    assert list(graphs)[0].startswith("G00_") and list(graphs)[-1].startswith("G08_")
    assert list(heads)[0].startswith("H00_") and list(heads)[-1].startswith("H11_")


def test_nonfinite_incomplete_head_summary_is_strict_json_safe():
    assert finite_float_or_none(np.nan) is None
    assert finite_float_or_none(np.inf) is None
    assert finite_float_or_none(3.25) == 3.25
    json.dumps({"mean_runtime_seconds": finite_float_or_none(np.nan)}, allow_nan=False)


def test_knn_lexical_tie_and_union_mutual():
    x = np.array([[0., 0.], [1., 0.], [-1., 0.], [5., 0.], [6., 0.]])
    ids = np.array(["m", "z", "a", "q", "p"])
    nn = _neighbors(x, 1, "euclidean", ids)
    assert nn[0, 0] == 2  # equal distance: barcode a before z
    union = binary_knn(x, 1, "euclidean", ids, "union")
    mutual = binary_knn(x, 1, "euclidean", ids, "mutual")
    assert union.nnz > mutual.nnz
    assert np.all(union.diagonal() == 0) and np.all(mutual.diagonal() == 0)


def test_k_values_and_no_metric_fallback():
    rng = np.random.default_rng(10)
    x = rng.normal(size=(25, 5)); ids = [f"id{i:02d}" for i in range(25)]
    for k in (3, 6, 10, 18):
        assert _neighbors(x, k, "euclidean", ids).shape == (25, k)
    assert _neighbors(x, 10, "correlation", ids).shape == (25, 10)
    with pytest.raises(ValueError):
        _neighbors(x, 10, "cosine", ids)


def test_per_modality_intersection_and_isolates():
    spatial = sp.csr_matrix(np.array([[0,1,1,0],[1,0,0,0],[1,0,0,1],[0,0,1,0]]))
    f1 = sp.csr_matrix(np.array([[0,1,0,0],[1,0,0,0],[0,0,0,0],[0,0,0,0]]))
    f2 = sp.csr_matrix(np.array([[0,0,1,0],[0,0,0,0],[1,0,0,1],[0,0,1,0]]))
    s1, s2 = spatial.multiply(f1), spatial.multiply(f2)
    assert s1.nnz == 2 and s2.nnz == 4 and (s1 != s2).nnz > 0
    norm = normalize_support(s1)
    assert np.all(norm.diagonal() > 0)  # isolates receive normalization self-loop only
    scores = moran_scores(np.arange(12, dtype=float).reshape(4, 3), s1)
    assert scores.shape == (3,) and np.isfinite(scores).all()


def test_reliability_and_self_tuning_affinity():
    rng = np.random.default_rng(20)
    views = [rng.normal(size=(32, 5)) for _ in range(3)]
    weights = reliability_weights(views, 10)
    assert np.isclose(weights.sum(), 1.0) and np.all(weights >= 0)
    affinity = self_tuning_affinity(views[0], 10)
    assert sp.issparse(affinity) and affinity.nnz > 0
    assert np.max(np.abs((affinity - affinity.T).data), initial=0.0) == 0
    assert np.all(affinity.data > 0) and np.all(affinity.data <= 1)


@pytest.mark.parametrize("head_id", [row["id"] for row in REG["head_candidates"]])
def test_all_heads_execute_without_labels(head_id, tmp_path):
    rng = np.random.default_rng(2020)
    # Three visible Gaussian groups make every registered head well-defined.
    labels = np.repeat(np.arange(3), 30)
    centers = np.zeros((3, 6), dtype=np.float64)
    centers[0, 0] = 3.0; centers[1, 1] = 3.0; centers[2, 2] = 3.0
    base = rng.normal(scale=.2, size=(90, 6)) + centers[labels]
    views = {"emb_latent_omics1": base + rng.normal(scale=.05, size=base.shape),
             "emb_latent_omics2": base + rng.normal(scale=.05, size=base.shape),
             "SpaLORA_fused": base + rng.normal(scale=.03, size=base.shape)}
    coords = np.column_stack((np.arange(90) % 15, np.arange(90) // 15))
    ids = np.asarray([f"spot{i:03d}" for i in range(90)])
    head = {row["id"]: row for row in REG["head_candidates"]}[head_id]
    observed, audit = run_head(head, views, 3, coords, ids, tmp_path / head_id)
    assert len(observed) == 90 and len(np.unique(observed)) == 3
    assert audit["fallback"] is False


def test_firewall_negative_cases():
    safe = "/root/autodl-fs/night6b_data_20260817/tonsil_label_free/tonsil_s1_rna_label_free.h5ad"
    assert guard_path(safe, role="trainer_transformer", operation="read")
    for path in ("/root/autodl-fs/datasets/human_tonsil_official/section1/s1_adata_rna.h5ad",
                 "/root/autodl-fs/P22/data.h5ad", "/root/autodl-fs/Human lymph node/D1/data.h5ad",
                 "/root/autodl-fs/GSE198353/data.h5ad", "/root/autodl-fs/night4b_raw/x"):
        with pytest.raises(FirewallViolation):
            guard_path(path, role="trainer_transformer", operation="read")
    with pytest.raises(FirewallViolation):
        reject_transform_payload(labels=np.array([1, 2]))
    with pytest.raises(FirewallViolation):
        reject_transform_payload(intermediate_epoch_metric=.5)


def test_two_step_toy_checkpoint_fresh_process(tmp_path):
    rng = np.random.default_rng(4)
    n = 12
    idx = torch.arange(n, dtype=torch.long)
    identity = torch.sparse_coo_tensor(torch.stack((idx, idx)), torch.ones(n), (n, n)).coalesce()
    data = {"features_omics1": rng.normal(size=(n, 5)).astype(np.float32),
            "features_omics2": rng.normal(size=(n, 4)).astype(np.float32),
            "adj_spatial_omics1": identity, "adj_feature_omics1": identity,
            "adj_spatial_omics2": identity, "adj_feature_omics2": identity}
    cfg = {"embedding_dim": 3, "epochs": 2, "loss_factors": [1.9,2.5,1.5,10.],
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
    subprocess.run([sys.executable, "-c", code, str(checkpoint), str(output)],
                   cwd=REPO, check=True)
    with np.load(output) as fresh:
        for key, value in expected.items():
            assert np.allclose(value, fresh[key], atol=1e-6, rtol=1e-5)
