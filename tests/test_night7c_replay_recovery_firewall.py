from pathlib import Path

import pytest

from SpaLORA.night7c_firewall import LabelFirewall, LabelFirewallError, reject_low_level_obs_read


@pytest.mark.parametrize("name", ["dataset", "tissue", "organism", "platform", "modality", "filename", "ari", "nmi", "q"])
def test_router_identity_and_metric_keys_are_rejected(name):
    allowed = {"m_initial", "conflict", "rank_c", "quality", "support"}
    supplied = allowed | {name}
    assert supplied - allowed == {name}


def test_recovery_prelock_rejects_original_h5ad_obs_labels_and_metrics(tmp_path):
    locked = tmp_path / "locked_arrays.npz"
    denied = tmp_path / "authority_labels"
    firewall = LabelFirewall([locked], [denied])
    firewall.require_modeling_read(locked)
    for bad in (tmp_path / "original.h5ad", denied / "labels.csv", denied / "metrics.csv"):
        with pytest.raises(LabelFirewallError):
            firewall.require_modeling_read(bad)
    with pytest.raises(LabelFirewallError):
        reject_low_level_obs_read(tmp_path / "original.h5ad", "obs/final_annot")


def test_recovery_has_single_label_window_and_no_return(tmp_path):
    denied = tmp_path / "labels"
    firewall = LabelFirewall([tmp_path / "locked.npz"], [denied])
    firewall.authorize_locked_outputs()
    firewall.open_evaluator_window()
    firewall.require_label_read(denied / "truth.csv")
    with pytest.raises(LabelFirewallError):
        firewall.open_evaluator_window()
    with pytest.raises(LabelFirewallError):
        firewall.require_transform_allowed()
