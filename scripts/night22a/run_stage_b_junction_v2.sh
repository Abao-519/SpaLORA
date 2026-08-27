#!/usr/bin/env bash
set -euo pipefail
cd /root/SpaLORA-night16h
# Reuse the audited lane/authority table from cycle 1 while changing only the
# active scientific registry and output namespace.  The sed output is executed
# directly and does not mutate the cycle-1 script or artifacts.
bash <(
  sed \
    -e 's|REG=configs/night22a/stage_b_junction_freeze.json|REG=configs/night22a/stage_b_junction_freeze_v2.json|' \
    -e 's|OUT=/root/night22a_working/junction$|OUT=/root/night22a_working/junction_v2|' \
    -e 's|EVAL=/root/night22a_working/junction_evaluations$|EVAL=/root/night22a_working/junction_evaluations_v2|' \
    -e 's|REPLAY=/root/night22a_working/junction_replays$|REPLAY=/root/night22a_working/junction_replays_v2|' \
    scripts/night22a/run_stage_b_junction.sh
)
