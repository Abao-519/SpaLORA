from __future__ import annotations

import inspect
import os
import signal
import subprocess
import sys
import time


def test_resource_bounded_worker_reuses_exact_transform_call() -> None:
    from scripts import night7c_stagew_resource_bounded as bounded

    source = inspect.getsource(bounded.worker_main)
    assert 'transform_cell(payload["training"], payload["unit"])' in source
    assert bounded.WORKERS == 4
    assert bounded.W00_LIMIT_SECONDS == 16 * 3600
    assert bounded.UNIT_LIMIT_SECONDS == 60 * 60
    assert bounded.CONTINUATION_LIMIT_SECONDS == 12 * 3600


def test_synthetic_timeout_kills_only_isolated_process_group() -> None:
    from scripts import night7c_stagew_resource_bounded as bounded

    sentinel = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"], start_new_session=True)
    target = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import subprocess,sys,time;"
                "subprocess.Popen([sys.executable,'-c','import time;time.sleep(300)']);"
                "time.sleep(300)"
            ),
        ],
        start_new_session=True,
    )
    try:
        time.sleep(0.2)
        identity = bounded.proc_identity(target.pid)
        assert identity["pgid"] == target.pid
        assert identity["sid"] == target.pid
        started = time.monotonic()
        os.killpg(identity["pgid"], signal.SIGSTOP)
        for _ in range(200):
            if bounded.proc_identity(target.pid)["state"] in {"T", "t"}:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("synthetic group did not stop")
        result = bounded.stop_exact_group(
            target.pid,
            identity["start_ticks"],
            identity["pgid"],
            already_stopped=True,
            grace_seconds=0.2,
        )
        assert result["terminated_by"] in {"SIGTERM", "SIGKILL_AFTER_GRACE"}
        target.wait(timeout=5)
        assert time.monotonic() - started < 5
        assert sentinel.poll() is None
    finally:
        for process in (target, sentinel):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
