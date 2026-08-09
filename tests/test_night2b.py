import glob
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from SpaLORA.model import AttentionLayer
from SpaLORA.night2b_loss_audit import (
    LOSS_LOG_FIELDS,
    legacy_bug_weight_vector,
    required_checkpoint_epochs,
    rna_loss_dispatch,
    validate_name_weight_alignment,
)


REPO = Path(__file__).resolve().parents[1]
PROTECTED = Path("/root/autodl-fs/night2b_preexisting_20260809/protected_before.sha256")


def test_bad_weight_mean_is_exact_for_frozen_dimensions():
    expected = {2000: 2.290322960, 3000: 2.289938091}
    for genes, target in expected.items():
        features = torch.arange(2 * genes, dtype=torch.float32).reshape(2, genes)
        value = legacy_bug_weight_vector(features)
        assert abs(float(value.mean()) - target) <= 1e-6
        assert torch.all(value >= 1.0) and torch.all(value <= 6.0)


def test_v3_replays_frozen_reduction_and_identity():
    features = torch.arange(6000, dtype=torch.float32).reshape(2, 3000)
    diff = torch.linspace(-2.0, 3.0, 6000).reshape(2, 3000)
    result = rna_loss_dispatch(diff, features, "locked_legacy_loss_replay")
    frozen = torch.mean((diff ** 2) * legacy_bug_weight_vector(features).unsqueeze(0))
    assert torch.equal(result.final, frozen)
    assert torch.allclose(result.final, result.m_bad * result.weighted_before_global, atol=1e-6, rtol=1e-6)


def test_v1_is_uniform_shape_with_only_mechanical_global_scale():
    features = torch.arange(6000, dtype=torch.float32).reshape(2, 3000)
    diff = torch.linspace(-1.0, 1.0, 6000).reshape(2, 3000)
    result = rna_loss_dispatch(diff, features, "locked_uniform_legacy_scale")
    assert torch.equal(result.gene_weights, torch.ones_like(result.gene_weights))
    assert torch.allclose(result.final, result.m_bad * result.raw_unweighted)


def test_v4_name_weight_order_guard():
    names = np.asarray(["a", "b", "c"])
    weights = validate_name_weight_alignment(names, names.copy(), [1.1, 1.2, 1.3])
    assert weights.dtype == np.float32 and weights.shape == (3,)
    with pytest.raises(AssertionError):
        validate_name_weight_alignment(names, names[::-1], [1.1, 1.2, 1.3])


def test_loss_log_checkpoints_and_schema_are_frozen():
    assert required_checkpoint_epochs(200) == [0, 20, 100, 199]
    assert required_checkpoint_epochs(1600) == [0, 160, 800, 1599]
    assert len(LOSS_LOG_FIELDS) == len(set(LOSS_LOG_FIELDS))


def test_legacy_attention_rows_sum_to_one():
    attention = AttentionLayer(4, 4)
    _, alpha = attention(torch.randn(7, 4), torch.randn(7, 4))
    assert alpha.shape == (7, 2)
    assert torch.allclose(alpha.sum(dim=1), torch.ones(7), atol=1e-7)


def test_p0b_authorization_and_all_dataset_checks():
    report_path = REPO / "reports/night2b_parity_locked.json"
    if not report_path.exists():
        pytest.skip("P0B target audit has not run yet")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    gate = json.loads((REPO / "results/night2b/gate_status.json").read_text(encoding="utf-8"))
    assert report["ground_truth_accessed"] is False
    assert report["p0b_pass"] is True
    assert gate["factorial_authorized"] is True
    assert all(item["consumed_inputs"]["pass"] for item in report["datasets"].values())
    assert all(item["initial_model_state"]["pass"] for item in report["datasets"].values())
    assert all(item["v3_one_adam_step"]["pass"] for item in report["datasets"].values())
    assert all(item["v3_five_step_trajectory"]["pass"] for item in report["datasets"].values())


def test_labels_are_loaded_only_after_saved_unsupervised_predictions():
    import scripts.night2b_loss_audit as runner

    source = inspect.getsource(runner.run_one)
    first_embedding_save = source.index("save_prediction_artifacts")
    cluster = source.index("cluster_exact")
    cluster_save = source.index("save_prediction_artifacts", first_embedding_save + 1)
    labels = source.index("load_evaluation_labels")
    assert first_embedding_save < cluster < cluster_save < labels


def test_authorized_result_completeness():
    gate_path = REPO / "results/night2b/gate_status.json"
    if not gate_path.exists():
        pytest.skip("P0B target audit has not run yet")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("factorial_authorized") is not True:
        assert glob.glob(str(REPO / "results/night2b/raw/*/*/seed_*/metrics.json")) == []
        return
    main = glob.glob(str(REPO / "results/night2b/raw/*/*/seed_*/metrics.json"))
    tutorials = glob.glob(str(REPO / "results/night2b/tutorial2022/*/metrics.json"))
    assert len(main) == 75
    assert len(tutorials) == 3
    for metrics in main:
        parent = Path(metrics).parent
        assert all((parent / name).is_file() for name in ("clusters.csv", "attention.npz", "embedding.npz", "loss_components.csv"))


def test_all_preexisting_night1_night2_and_frozen_files_are_unchanged():
    if not PROTECTED.exists():
        pytest.skip("Target protected-file checksum manifest is unavailable")
    for line in PROTECTED.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(None, 1)
        relative = relative.strip()
        if relative == ".gitignore":
            continue  # Night-2B adds only its raw/cache ignore rules here.
        path = REPO / relative
        assert path.is_file(), relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative
