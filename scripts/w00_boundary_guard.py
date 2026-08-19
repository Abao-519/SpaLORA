#!/usr/bin/env python3
"""Stop one exact Stage-W driver immediately after W00's atomic manifest appears."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path


def proc_identity(pid: int) -> dict:
    stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    close = stat_text.rfind(")")
    if close < 0:
        raise RuntimeError("malformed /proc stat")
    prefix = stat_text[: close + 1]
    fields = stat_text[close + 2 :].split()
    return {
        "pid": int(prefix.split(" ", 1)[0]),
        "state": fields[0],
        "ppid": int(fields[1]),
        "pgid": int(fields[2]),
        "sid": int(fields[3]),
        "start_ticks": int(fields[19]),
        "cmdline": Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode().strip(),
    }


def append_event(log_path: Path, event: str, **values: object) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **values,
    }
    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--start-ticks", type=int, required=True)
    parser.add_argument("--pgid", type=int, required=True)
    parser.add_argument("--cmdline", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=0.02)
    args = parser.parse_args()
    expected = {
        "pid": args.pid,
        "start_ticks": args.start_ticks,
        "pgid": args.pgid,
        "cmdline": args.cmdline,
        "manifest": str(args.manifest),
    }

    guard_pid = os.getpid()
    guard_pgid = os.getpgrp()
    if guard_pgid == args.pgid:
        append_event(args.log, "guard_refused_same_pgid", guard_pid=guard_pid, guard_pgid=guard_pgid)
        return 20

    try:
        initial = proc_identity(args.pid)
    except Exception as exc:
        append_event(args.log, "guard_initial_identity_error", error=repr(exc), driver_pid=args.pid)
        return 21
    if (
        initial["start_ticks"] != args.start_ticks
        or initial["pgid"] != args.pgid
        or initial["cmdline"] != args.cmdline
    ):
        append_event(args.log, "guard_initial_identity_mismatch", expected=expected, actual=initial)
        return 22

    append_event(
        args.log,
        "guard_armed",
        driver=initial,
        guard_pid=guard_pid,
        guard_pgid=guard_pgid,
        guard_sid=os.getsid(0),
        manifest=str(args.manifest),
        poll_seconds=args.poll_seconds,
        label_access=False,
        scientific_computation=False,
    )

    while True:
        if args.manifest.is_file():
            # atomic_json publishes this path only after close, fsync and os.replace.
            try:
                with args.manifest.open("rb") as handle:
                    os.fstat(handle.fileno())
            except Exception as exc:
                append_event(args.log, "manifest_open_error", error=repr(exc))
                time.sleep(args.poll_seconds)
                continue
            try:
                current = proc_identity(args.pid)
            except Exception as exc:
                append_event(args.log, "driver_missing_at_manifest_boundary", error=repr(exc))
                return 23
            if (
                current["start_ticks"] != args.start_ticks
                or current["pgid"] != args.pgid
                or current["cmdline"] != args.cmdline
            ):
                append_event(args.log, "identity_mismatch_at_manifest_boundary", expected=expected, actual=current)
                return 24
            os.kill(args.pid, signal.SIGSTOP)
            append_event(
                args.log,
                "sigstop_sent",
                driver=current,
                manifest=str(args.manifest),
                manifest_size=args.manifest.stat().st_size,
                signal="SIGSTOP",
            )
            for _ in range(500):
                try:
                    stopped = proc_identity(args.pid)
                except Exception as exc:
                    append_event(args.log, "driver_disappeared_after_sigstop", error=repr(exc))
                    return 25
                if stopped["state"] in {"T", "t"}:
                    append_event(args.log, "driver_stop_confirmed", driver=stopped)
                    return 0
                time.sleep(0.01)
            append_event(args.log, "driver_stop_not_confirmed", driver=proc_identity(args.pid))
            return 26

        try:
            current = proc_identity(args.pid)
        except Exception as exc:
            append_event(args.log, "driver_exited_before_manifest", error=repr(exc))
            return 27
        if current["start_ticks"] != args.start_ticks:
            append_event(args.log, "pid_reuse_before_manifest", expected_start=args.start_ticks, actual=current)
            return 28
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
