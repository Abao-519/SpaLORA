from pathlib import Path

import pytest

from SpaLORA.night6b_firewall import FirewallViolation, guard_path, reject_label_payload


ORIGINAL = "/root/autodl-fs/datasets/human_tonsil_official/section1/s1_adata_rna.h5ad"


def test_trainer_cannot_open_original_tonsil():
    with pytest.raises(FirewallViolation):
        guard_path(ORIGINAL, role="trainer_transformer", operation="anndata.read_h5ad")


def test_data_steward_can_low_level_open_original_tonsil():
    assert guard_path(ORIGINAL, role="data_steward", operation="h5py.File") == Path(ORIGINAL)


def test_prelock_evaluator_cannot_open_original_tonsil():
    with pytest.raises(FirewallViolation):
        guard_path(ORIGINAL, role="evaluator", operation="label_read", phase_locked=False)


@pytest.mark.parametrize("path", [
    "/root/autodl-fs/P22/data.h5ad",
    "/root/autodl-fs/Human_Lymph_Node_D1/data.h5ad",
    "/root/autodl-fs/GSE198353/data.h5ad",
    "/root/autodl-fs/night6a_raw_runs_20260814/a1/run_manifest.json",
])
def test_protected_paths_fail_closed(path):
    with pytest.raises(FirewallViolation):
        guard_path(path, role="trainer_transformer", operation="open")


def test_candidate_transform_rejects_label_payload():
    with pytest.raises(FirewallViolation):
        reject_label_payload(label_vector=[1, 2, 3])


def test_candidate_transform_rejects_metric_payload():
    with pytest.raises(FirewallViolation):
        reject_label_payload(ari=0.5)

