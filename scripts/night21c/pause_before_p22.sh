#!/usr/bin/env bash
set -euo pipefail
parent_pid="$1"
while kill -0 "$parent_pid" 2>/dev/null; do
  line="$(ps -o cmd= --ppid "$parent_pid" || true)"
  if [[ "$line" == *"TONSIL_S1_K4"*"OFFICIAL_GRAPH_ONLY"* ]]; then
    kill -STOP "$parent_pid"
    echo "STOPPED_PARENT_BEFORE_P22 parent=$parent_pid"
    exit 0
  fi
  sleep 3
done
echo "PARENT_EXITED_BEFORE_STOP parent=$parent_pid"
