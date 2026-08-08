import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from SpaLORA.model_corrected import AttentionCorrected
from SpaLORA.night1_pipeline import normalize_graph_sparse
from SpaLORA.night2_loss_audit import (
    LOSS_LOG_FIELDS,
    LossDynamicsRecorder,
    legacy_bug_weight_vector,
    loss_decomposition,
    required_checkpoint_epochs,
)


REPO = Path(__file__).resolve().parents[1]


def test_legacy_bug_weight_mean_matches_mechanical_expected_values():
    expected = {2000: 2.290322960, 3000: 2.289938091}
    for genes, target in expected.items():
        features = torch.arange(2 * genes, dtype=torch.float32).reshape(2, genes)
        weights = legacy_bug_weight_vector(features)
        assert abs(float(weights.mean()) - target) <= 1e-6
        assert torch.all(weights >= 1) and torch.all(weights <= 6)


def test_v3_identity_is_exact_within_float_tolerance():
    per_gene = torch.linspace(0.1, 3.0, 3000)
    features = torch.arange(6000, dtype=torch.float32).reshape(2, 3000)
    weights = legacy_bug_weight_vector(features)
    values = loss_decomposition(per_gene, weights)
    assert torch.allclose(
        values["legacy_loss_replay"],
        values["m_bad"] * values["legacy_shape_normalized"],
        atol=1e-6,
        rtol=1e-6,
    )


def test_v1_has_uniform_gene_shape_and_only_global_scale():
    per_gene = torch.linspace(1.0, 8.0, 3000)
    features = torch.arange(6000, dtype=torch.float32).reshape(2, 3000)
    values = loss_decomposition(per_gene, legacy_bug_weight_vector(features))
    assert torch.allclose(values["uniform_legacy_scale"], values["m_bad"] * per_gene.mean())


def test_loss_logging_schema_and_checkpoints_are_complete():
    recorder = LossDynamicsRecorder(epochs=200)
    assert required_checkpoint_epochs(200) == [0, 20, 100, 199]
    for epoch in required_checkpoint_epochs(200):
        record = {name: 1.0 for name in LOSS_LOG_FIELDS}
        record["epoch"] = epoch
        recorder.add(record)
    recorder.assert_complete()


def test_attention_rows_sum_to_one_on_explicit_axis():
    layer = AttentionCorrected(4, 4)
    _, alpha = layer(torch.randn(7, 4), torch.randn(7, 4))
    assert alpha.shape == (7, 2)
    assert torch.allclose(alpha.sum(dim=1), torch.ones(7), atol=1e-7)


def test_sparse_graph_normalization_remains_sparse():
    import scipy.sparse as sp

    graph = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32))
    normalized = normalize_graph_sparse(graph)
    assert normalized.is_sparse
    assert normalized._nnz() == 7


def test_ground_truth_guard_from_night1_remains_in_force():
    import inspect
    import scripts.night1_benchmark as benchmark

    source = inspect.getsource(benchmark.run_one)
    label_position = source.index("load_evaluation_labels")
    assert source.index("trainer.train()") < label_position
    assert source.index("cluster_exact") < label_position


def test_p0_failure_prevents_factorial_outputs():
    report = json.loads((REPO / "reports/night2_parity.json").read_text(encoding="utf-8"))
    assert report["p0_pass"] is False
    assert report["factorial_authorized"] is False
    assert glob.glob(str(REPO / "results/night2/raw/*/*/seed_*/metrics.json")) == []


def test_p0_stopped_run_outputs_are_explicit_and_complete():
    status = json.loads((REPO / "results/night2/gate_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "stopped_by_p0_hard_gate"
    assert status["new_main_runs"] == {"expected_if_authorized": 60, "completed": 0, "skipped": 60}
    required = (
        "per_seed_metrics.csv", "summary.csv", "paired_deltas.csv", "factorial_effects.csv",
        "loss_components.csv", "paper_repro_audit.csv", "attention_summary.csv", "per_domain_f1.csv",
    )
    assert all((REPO / "results/night2" / name).is_file() for name in required)


def test_night1_protected_files_match_pre_run_checksums():
    manifest = Path("/root/autodl-fs/night2_preexisting_20260809/night1_before.sha256")
    if not manifest.exists():
        import pytest
        pytest.skip("Target-machine pre-run Night-1 checksum manifest is not available")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(None, 1)
        path = REPO / relative.strip()
        assert path.is_file(), relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative
