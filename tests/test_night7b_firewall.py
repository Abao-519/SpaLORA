from pathlib import Path


def test_training_worker_has_no_label_reader_or_evaluator_import():
    text = (Path(__file__).resolve().parents[1] / "scripts/night7b_train.py").read_text().lower()
    assert "import anndata" not in text
    assert "read_h5ad" not in text
    assert "night7b_evaluate" not in text
    assert "ground_truth" in text  # fail-closed forbidden-key declaration


def test_training_module_has_no_metric_implementation():
    text = (Path(__file__).resolve().parents[1] / "SpaLORA/night7b_adaptive.py").read_text().lower()
    assert "adjusted_rand" not in text
    assert "normalized_mutual" not in text
