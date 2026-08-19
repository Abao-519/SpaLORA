from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from SpaLORA.night8a_mfspc import (
    EMAScaler, MFSPCModel, centered_cross_covariance_loss, dgi_loss,
    fixed_triplets, normalized_assays, prototype_loss, resolve_modules,
    rna_anchor_support, select_family, tensor_state_sha, vicreg_loss,
)


class ExplodingSentinel:
    def __getattribute__(self, name):
        raise AssertionError("forbidden sentinel accessed")
    def __iter__(self):
        raise AssertionError("forbidden sentinel iterated")
    def __repr__(self):
        raise AssertionError("forbidden sentinel represented")


@pytest.mark.parametrize("assays,expected", [
    (["RNA", "ADT"], "RNA_PROTEIN"),
    (["RNA", "protein"], "RNA_PROTEIN"),
    (["transcriptome", "proteomics"], "RNA_PROTEIN"),
    (["RNA", "ATAC"], "RNA_EPIGENOME"),
    (["RNA", "histone"], "RNA_EPIGENOME"),
    (["RNA", "epigenome"], "RNA_EPIGENOME"),
    (["gene expression", "chromatin accessibility"], "RNA_EPIGENOME"),
])
def test_family_positive(assays, expected):
    assert select_family({"assays": assays}, ExplodingSentinel()) == expected


@pytest.mark.parametrize("metadata", [
    {}, {"assays": ["RNA"]}, {"assays": ["RNA", "lipid"]},
    {"assays": "RNA+ADT"}, {"assays": ["ADT", "protein"]},
])
def test_family_fail_closed(metadata):
    with pytest.raises(ValueError):
        select_family(metadata)


@pytest.mark.parametrize("identity", ["A1", "D1", "tonsil", "P22", "renamed-opaque-491"])
def test_identity_rename_invariance(identity):
    assert select_family({"assays": ["RNA", "ADT"], "display_name": identity}) == "RNA_PROTEIN"


def test_assay_normalization_order_invariant():
    assert normalized_assays({"assays": ["ADT", "RNA"]}) == normalized_assays({"assays": ["RNA", "ADT"]})


def test_rna_anchor_noop_protein():
    assert resolve_modules(["SP", "RNA_ANCHOR"], "RNA_PROTEIN") == ("SP",)


def test_rna_anchor_active_epigenome():
    assert resolve_modules(["SP", "RNA_ANCHOR"], "RNA_EPIGENOME") == ("SP", "RNA_ANCHOR")


@pytest.mark.parametrize("modules", [["RR10", "RR30"], ["BOGUS"], ["SP", "SP"]])
def test_module_registry_fail_closed(modules):
    with pytest.raises(ValueError):
        resolve_modules(modules, "RNA_PROTEIN")


def inputs(n=24, d=8):
    torch.manual_seed(3)
    x1 = torch.randn(n, d); x2 = torch.randn(n, d); ref = torch.nn.functional.normalize(torch.randn(n, d), dim=1)
    return x1, x2, ref


def test_module_off_exact_reference_parity():
    x1, x2, ref = inputs(); model = MFSPCModel(8, [], 4)
    out = model(x1, x2, ref)
    assert torch.equal(out["fused"], ref)


def test_distinct_input_and_reference_dimensions():
    torch.manual_seed(4)
    model = MFSPCModel(16, ["SP", "RR10"], 4, fused_dim=8)
    out = model(torch.randn(20, 16), torch.randn(20, 16),
                torch.nn.functional.normalize(torch.randn(20, 8), dim=1))
    assert out["recon1"].shape == (20, 16) and out["fused"].shape == (20, 8)
    assert out["shared1"].shape[1] == 6 and out["private1"].shape[1] == 2


def test_sp_dimensions_and_gradients():
    x1, x2, ref = inputs(); model = MFSPCModel(8, ["SP"], 4)
    out = model(x1, x2, ref)
    assert out["shared1"].shape[1] == 6 and out["private1"].shape[1] == 2
    loss = (centered_cross_covariance_loss(out["shared1"], out["private1"])
            + centered_cross_covariance_loss(out["shared2"], out["private2"])
            + out["recon1"].square().mean() + out["recon2"].square().mean()
            + out["fused"].square().mean())
    loss.backward(); assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_vicreg_finite_backward():
    a = torch.randn(32, 8, requires_grad=True); b = torch.randn(32, 8, requires_grad=True)
    loss, detail = vicreg_loss(a, b); loss.backward()
    assert torch.isfinite(loss) and len(detail) == 3 and torch.isfinite(a.grad).all()


def test_prototype_finite_backward_and_teacher_update():
    x1, x2, ref = inputs(); model = MFSPCModel(8, ["PROTO"], 4)
    out = model(x1, x2, ref); loss, _, occ = prototype_loss(model, out)
    loss.backward(); before = model.teacher_prototypes.clone(); model.update_teacher(occ)
    assert torch.isfinite(loss) and not torch.equal(before, model.teacher_prototypes)


def test_dgi_sparse_style_finite():
    local = torch.randn(20, 6, requires_grad=True); perm = torch.arange(19, -1, -1)
    loss = dgi_loss(local, perm); loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(local.grad).all()


def test_rna_anchor_sparse_intersection_self_loops():
    rng = np.random.default_rng(2); n = 30
    spatial = sp.diags([np.ones(n - 1), np.ones(n - 1)], [-1, 1], shape=(n, n)).tocsr()
    result = rna_anchor_support(spatial, rng.normal(size=(n, 6)), [f"x{i}" for i in range(n)], 4)
    assert sp.issparse(result) and result.nnz < n * n and np.all(result.diagonal() > 0)


def test_fixed_triplets_contract_and_determinism():
    rng = np.random.default_rng(4); a = rng.normal(size=(40, 8)); b = rng.normal(size=(40, 8)); ids = [f"s{i:03d}" for i in range(40)]
    x = fixed_triplets(a, b, ids, 5); y = fixed_triplets(a, b, ids, 5)
    assert np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1])
    assert x[2]["k"] == 3 and x[2]["farthest_fraction"] == .60 and not x[2]["resident_dense_n_by_n"]


def test_ema_scale_warmup_and_clip():
    scale = EMAScaler(); base = torch.tensor(10.0); aux = torch.tensor(.001)
    assert scale.factor("x", base, aux, 0) == 0.0
    assert scale.factor("x", base, aux, 10) == 10.0


def test_tensor_state_sha_deterministic_and_sensitive():
    model = MFSPCModel(8, ["SP"], 3); first = tensor_state_sha(model.state_dict())
    assert first == tensor_state_sha(model.state_dict())
    with torch.no_grad(): next(model.parameters()).add_(1)
    assert first != tensor_state_sha(model.state_dict())


def test_registry_has_exact_locked_candidates():
    path = Path("protocols/night8a/SpaLORA_Night8A_MFSPC_Registry_2026-08-20.json")
    registry = json.loads(path.read_text())
    assert [x["id"] for x in registry["R1_configs"]] == [f"A0{i}_{name}" for i, name in enumerate([
        "FAMILY_REFERENCE", "SP", "RR10", "RR30", "PROTO", "RNA_ANCHOR", "DGI", "SMART_TRIPLET"])]
    assert [x["id"].split("_")[0] for x in registry["R2_configs"]] == [f"B0{i}" for i in range(8)]


@pytest.mark.parametrize("key,value", [
    ("science_training_hard_cap", 180), ("infrastructure_retry_hard_cap", 12),
    ("scientific_retry_hard_cap", 0), ("fallback_hard_cap", 0),
    ("cpu_transform_workers_max", 3), ("threads_per_worker_max", 3),
    ("single_transform_wall_timeout_minutes", 45),
])
def test_registry_runtime_locks(key, value):
    registry = json.loads(Path("protocols/night8a/SpaLORA_Night8A_MFSPC_Registry_2026-08-20.json").read_text())
    assert registry["runtime"][key] == value
