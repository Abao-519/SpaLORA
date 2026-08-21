from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.night9c_p0_data import ANNOTATION_KEY, copy_label_free, schema_contract


def test_low_level_copy_removes_all_obs_columns(tmp_path: Path) -> None:
    source = tmp_path / "source.h5ad"
    target = tmp_path / "safe.h5ad"
    obj = ad.AnnData(
        X=sp.csr_matrix(np.asarray([[1, 0, 2], [0, 3, 0]], dtype=np.int32)),
        obs=pd.DataFrame({ANNOTATION_KEY: ["secret_a", "secret_b"], "qc": [1, 2]}, index=["s0", "s1"]),
        var=pd.DataFrame({"kind": ["a", "b", "c"]}, index=["v0", "v1", "v2"]),
    )
    obj.obsm["spatial"] = np.asarray([[0.0, 1.0], [2.0, 3.0]])
    obj.uns["peaks"] = np.asarray(["p0", "p1"])
    obj.uns["label_colors"] = np.asarray(["red", "blue"])
    obj.write_h5ad(source)

    schema = schema_contract(source)
    assert schema["annotation_path_present"] is True
    assert schema["annotation_values_read"] is False
    proof = copy_label_free(source, target)

    safe = ad.read_h5ad(target)
    assert proof["obs_columns_count"] == 0
    assert list(safe.obs.columns) == []
    assert list(safe.obs_names) == ["s0", "s1"]
    assert safe.shape == (2, 3)
    assert np.array_equal(safe.obsm["spatial"], obj.obsm["spatial"])
    assert "peaks" in safe.uns
    assert "label_colors" not in safe.uns
    assert np.array_equal(safe.X.toarray(), obj.X.toarray())
