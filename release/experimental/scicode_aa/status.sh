#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Show SciCode run progress/scores (runs the reporter on the worker).
exec ssh emi@10.100.32.2 /home/emi/code/ds41/artifacts/evals/scicode/.venv/bin/python /home/emi/code/ds41/artifacts/evals/scicode/status.py
