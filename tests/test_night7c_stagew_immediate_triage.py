import ast
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/night7c_stagew_immediate_triage.py"
DRIVER = Path(__file__).resolve().parents[1] / "scripts/night7c_stagew_immediate_triage_driver.py"


def source_and_tree():
    source = SCRIPT.read_text(encoding="utf-8")
    return source, ast.parse(source)


def test_triage_is_label_free_and_never_calls_partition_or_eigensolver():
    source, tree = source_and_tree()
    triage = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "triage")
    banned = {"run_partition", "spectral", "eigsh", "eigs", "lobpcg", "SpectralClustering"}
    called = set()
    for node in ast.walk(triage):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)
    assert not (called & banned)
    assert "label_access" in source
    assert "TRIAGE_LIMIT_SECONDS = 30 * 60" in source


def test_bounded_continuation_contract_is_fixed():
    source, _ = source_and_tree()
    assert "WORKERS = 4" in source
    assert "UNIT_LIMIT_SECONDS = 60 * 60.0" in source
    assert "TOTAL_LIMIT_SECONDS = 12 * 60 * 60.0" in source
    assert '"scientific_retry": 0' in source
    assert '"fallback": 0' in source
    assert 'scripts/night7c_stagew_resource_bounded.py"), "worker"' in source
    assert "RESOURCE_CENSORED_USER_STOP_AFTER_EXTREME_LONGTAIL" in source
    assert "SKIPPED_CANDIDATE_RESOURCE_CIRCUIT_BREAKER" in source
    assert "24dba16c3aae027e392062b6c81f93732585e119f07e3b87e962e22fb3770a75" in source


def test_isolated_triage_driver_has_no_partition_call_and_fixed_bounds():
    source = DRIVER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    banned = {"run_partition", "spectral", "eigsh", "eigs", "lobpcg", "SpectralClustering"}
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)
    assert not (called & banned)
    assert "WORKERS = 4" in source
    assert "CELL_LIMIT_SECONDS = 5 * 60.0" in source
    assert "TOTAL_LIMIT_SECONDS = 30 * 60.0" in source
    assert '"OMP_NUM_THREADS": "1"' in source
    assert '"MKL_NUM_THREADS": "1"' in source
    assert '"OPENBLAS_NUM_THREADS": "1"' in source
    assert "PRECHECK_INFRASTRUCTURE_INVALID" in source
