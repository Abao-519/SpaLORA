from pathlib import Path

import pytest

from SpaLORA.night7c_firewall import (
    LabelFirewall,
    LabelFirewallError,
    reject_low_level_obs_read,
)


def test_prelock_allowlist_and_h5ad_fail_closed(tmp_path):
    allowed = tmp_path / "locked_arrays.npz"
    denied = tmp_path / "labels"
    firewall = LabelFirewall([allowed], [denied])
    firewall.require_modeling_read(allowed)
    with pytest.raises(LabelFirewallError):
        firewall.require_modeling_read(tmp_path / "original.h5ad")
    with pytest.raises(LabelFirewallError):
        firewall.require_modeling_read(denied / "ground_truth.csv")


def test_low_level_obs_and_early_evaluator_fail_closed(tmp_path):
    denied = tmp_path / "labels"
    firewall = LabelFirewall([tmp_path / "x.npz"], [denied])
    with pytest.raises(LabelFirewallError):
        reject_low_level_obs_read(tmp_path / "original.h5ad", "obs/cell_type")
    with pytest.raises(LabelFirewallError):
        firewall.require_label_read(denied / "labels.npz")


def test_single_window_and_no_return_to_transform(tmp_path):
    denied = tmp_path / "labels"
    firewall = LabelFirewall([tmp_path / "x.npz"], [denied])
    firewall.authorize_locked_outputs()
    firewall.open_evaluator_window()
    firewall.require_label_read(denied / "labels.npz")
    with pytest.raises(LabelFirewallError):
        firewall.open_evaluator_window()
    with pytest.raises(LabelFirewallError):
        firewall.require_transform_allowed()
