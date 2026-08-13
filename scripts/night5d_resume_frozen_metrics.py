#!/usr/bin/env python3
"""Resume only post-lock statistics from the already frozen 50-row metric table."""
import json, pathlib, sys
REPO=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.night5d_evaluate import resume_from_frozen_metrics
if __name__=='__main__':resume_from_frozen_metrics()
