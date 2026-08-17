import pytest

from SpaLORA.night7a_firewall import Night7AFirewall


def test_prelock_transform_and_byte_hash_are_allowed():
    guard = Night7AFirewall()
    guard.byte_hash(fresh=True)
    guard.transform("transformer")
    assert guard.development_label_reads == 0
    assert guard.fresh_label_reads == 0


def test_prelock_label_read_is_fail_closed():
    guard = Night7AFirewall()
    with pytest.raises(PermissionError):
        guard.read_development_label("transformer")


def test_evaluator_requires_both_total_locks_and_is_single_use():
    guard = Night7AFirewall()
    with pytest.raises(RuntimeError):
        guard.lock_transforms_and_preflight(359, True)
    guard.lock_transforms_and_preflight(360, True)
    guard.read_development_label("evaluator")
    with pytest.raises(PermissionError):
        guard.read_fresh_label("evaluator")
    guard.close_evaluator()
    with pytest.raises(PermissionError):
        guard.transform("transformer")
