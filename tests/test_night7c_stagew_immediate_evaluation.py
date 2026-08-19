from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_total_lock_includes_triage_and_resource_outcomes():
    source = (ROOT / "scripts/night7c_replay_recovery_total_lock.py").read_text(encoding="utf-8")
    assert "stagew_immediate_semantic_triage.json" in source
    assert "stagew_resource_bounded_plan_and_eligibility.json" in source
    assert '"RESOURCE_CENSORED_USER_STOP_AFTER_EXTREME_LONGTAIL":1' in source
    assert '"SKIPPED_CANDIDATE_RESOURCE_CIRCUIT_BREAKER":7' in source
    assert '"success":40' in source


def test_evaluator_opens_labels_only_after_total_lock_and_excludes_w00_metrics():
    source = (ROOT / "scripts/night7c_replay_recovery_evaluate.py").read_text(encoding="utf-8")
    assert source.index('total_lock=json.loads((OUT/"total_prelabel_lock.json")') < source.index("labels,audit=labels_once()")
    assert 'for candidate in ["R02_REFERENCE",*eligible_w]:' in source
    assert '"scientific_metrics_computed":False' in source
    assert 'eligible_w==["W01_QUALITY_SOFT"' in source
