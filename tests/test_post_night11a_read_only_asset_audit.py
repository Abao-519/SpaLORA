from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).parents[1] / "scripts" / "post_night11a_direction_reset" / "read_only_asset_audit.py"
SPEC = spec_from_file_location("read_only_asset_audit", SOURCE)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_peak_coordinate_parser_is_explicit_and_fail_closed():
    row = MODULE.peak_summary(["chr1-10-20", "chr2:30-40", "peak_without_coordinates"])
    assert row["coordinate_parseable"] == 2
    assert row["coordinate_unparseable"] == 1


def test_null_snapshot_comparison_is_exact():
    value = {"roots": [{"root": "/tmp/example", "file_count": 1, "total_file_bytes": 2}]}
    result = MODULE.compare_snapshots(value, value)
    assert result["status"] == "PASS"
    assert result["label_file_contents_opened"] is False


def test_identifier_hash_is_order_sensitive():
    assert MODULE.sha_text(["a", "b"]) != MODULE.sha_text(["b", "a"])
    assert isinstance(np.asarray([1]), np.ndarray)
