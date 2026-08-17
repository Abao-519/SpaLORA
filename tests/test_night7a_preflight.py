import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/night7a_preflight.py"
SPEC = importlib.util.spec_from_file_location("night7a_preflight", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_label_selection_scanner_synthetic_fixture(tmp_path):
    repo = tmp_path / "source"
    repo.mkdir()
    (repo / "train.py").write_text(
        "ground_truth = labels\nbest_ari = max(adjusted_rand_score(y, z) for z in runs)\n",
        encoding="utf-8",
    )
    observed = MODULE.source_scan(repo)
    assert observed["label_access"]
    assert observed["metric_or_checkpoint_selection"]


def test_metadata_downloader_refuses_declared_large_file(monkeypatch, tmp_path):
    class Response:
        headers = {"Content-Length": "1001"}
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self, size): return b""
    monkeypatch.setattr(MODULE.urllib.request, "urlopen", lambda *a, **k: Response())
    with pytest.raises(RuntimeError, match="large-file refusal"):
        MODULE.fetch_small("https://example.invalid/big", tmp_path / "x", maximum=1000)


def test_toy_adapter_is_label_free_and_not_benchmark():
    observed = MODULE.toy_smoke()
    assert observed["status"] == "PASS"
    assert observed["fixed_final_label_free_adapter"]["labels_supplied"] is False
    assert observed["formal_benchmark_runs"] == 0
