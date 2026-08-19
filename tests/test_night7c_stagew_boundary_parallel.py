from __future__ import annotations

import inspect
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def test_boundary_guard_stops_only_exact_bound_process(tmp_path: Path) -> None:
    from scripts import w00_boundary_guard as guard

    child = subprocess.Popen(
        [sys.executable, "-c", "import time\nwhile True: time.sleep(1)"],
        start_new_session=True,
    )
    try:
        identity = guard.proc_identity(child.pid)
        manifest = tmp_path / "transform_manifest.json"
        log = tmp_path / "guard.jsonl"
        watcher = subprocess.Popen(
            [
                sys.executable,
                str(Path(guard.__file__)),
                "--pid",
                str(child.pid),
                "--start-ticks",
                str(identity["start_ticks"]),
                "--pgid",
                str(identity["pgid"]),
                "--cmdline",
                str(identity["cmdline"]),
                "--manifest",
                str(manifest),
                "--log",
                str(log),
                "--poll-seconds",
                "0.01",
            ],
            start_new_session=True,
        )
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if log.exists() and "guard_armed" in log.read_text(encoding="utf-8"):
                break
            time.sleep(0.01)
        else:
            raise AssertionError("test guard did not arm")

        temporary = manifest.with_name(manifest.name + ".tmp")
        temporary.write_text(json.dumps({"status": "success"}) + "\n", encoding="utf-8")
        os.replace(temporary, manifest)
        assert watcher.wait(timeout=5) == 0
        assert guard.proc_identity(child.pid)["state"] in {"T", "t"}
        events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        assert [row["event"] for row in events] == [
            "guard_armed",
            "sigstop_sent",
            "driver_stop_confirmed",
        ]
        assert events[0]["guard_pgid"] != identity["pgid"]
    finally:
        try:
            os.kill(child.pid, signal.SIGKILL)
            os.kill(child.pid, signal.SIGCONT)
        except ProcessLookupError:
            pass
        child.wait(timeout=5)


def test_parallel_scheduler_reuses_exact_transform_function() -> None:
    from scripts import night7c_stagew_boundary_parallel as scheduler

    source = inspect.getsource(scheduler._parallel_transform)
    assert "return transform_cell(training, unit)" in source
    run_source = inspect.getsource(scheduler.run_remaining)
    assert 'max_workers=4' in run_source
    assert 'mp.get_context("spawn")' in run_source
    assert 'remaining_count": len(remaining)' in run_source
    assert 'require(len(remaining) == 47' in run_source
