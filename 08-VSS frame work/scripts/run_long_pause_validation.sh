#!/usr/bin/env bash
# Prepared by Tahereh Fahi. Fresh, gated validation; no publication or dashboard writes.
set -u

framework_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
workspace_dir="$(cd "${framework_dir}/.." && pwd)" || exit 1
cd "$workspace_dir" || exit 1
if [[ ! -x .venv/bin/python ]]; then
  printf 'Missing workspace Python environment: .venv/bin/python\n' >&2
  exit 1
fi

batch_stamp="$(date +%Y%m%d_%H%M%S)_$$"
batch_start="$(date +%s)"
printf 'Batch: %s\nStages: Video 5 40s, Video 4 40s, Video 5 full, Video 4 full.\n' "$batch_stamp"
printf 'Each stage runs only after the preceding stage passes its saved-output gate.\n'

check_saved_run() {
  .venv/bin/python - "$1" "$2" "$3" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
video = sys.argv[2]
scope = sys.argv[3]
root = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
expected_scope = "smoke_prefix" if scope == "40" else "full_video"
if root.get("run_scope") != expected_scope or root.get("stage_scope") != "all":
    raise SystemExit("Run scope does not match the requested validation")
if root.get("quality_failed_stages"):
    raise SystemExit("A quality gate failed: " + str(root["quality_failed_stages"]))
if any(stage.get("status") != "complete" or stage.get("reused")
       for stage in root["stages"].values()):
    raise SystemExit("A stage is incomplete or reused")
if root["stages"]["hud"].get("quality_status") != "passed":
    raise SystemExit("HUD quality gate did not pass")

def checked_artifact(stage_name, relative):
    stage = root["stages"][stage_name]
    path = run / stage["directory"] / relative
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != stage["outputs"].get(relative):
        raise SystemExit("Output hash mismatch: " + str(path))
    return path

inventory_path = checked_artifact("inventory", "run_manifest.json")
inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
transactions = int(inventory["counts"]["level_up_transactions"])
pause_path = checked_artifact("release", "gameplay_pauses.json")
pauses = json.loads(pause_path.read_text(encoding="utf-8"))["intervals"]
if transactions < 2 or len(pauses) < 2:
    raise SystemExit(
        f"{video} {scope}: only {transactions} transactions and {len(pauses)} pauses; "
        "stopping before a longer run"
    )
print(f"GATE PASSED: {video} {scope}, {transactions} transactions, "
      f"{len(pauses)} pauses, HUD QC passed, saved hashes match")
PY
}

run_check() {
  local video_key="$1" scope="$2" output config started exit_code
  output="${framework_dir}/runs/${video_key}_pause_validation_${scope}_${batch_stamp}"
  config="${framework_dir}/configs/${video_key}.json"
  started="$(date +%s)"
  printf '\nRUNNING: %s %s\nOutput: %s\nLog: %s.terminal.log\n' "$video_key" "$scope" "$output" "$output"
  if [[ "$scope" == "40" ]]; then
    caffeinate -i env PYTHONPATH="${framework_dir}/src" \
      PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
      .venv/bin/python -m vss_framework.cli run-video \
      --config "$config" --output "$output" --max-seconds 40 \
      --stages all --no-resume >"$output.terminal.log" 2>&1
  else
    caffeinate -i env PYTHONPATH="${framework_dir}/src" \
      PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
      .venv/bin/python -m vss_framework.cli run-video \
      --config "$config" --output "$output" \
      --stages all --no-resume >"$output.terminal.log" 2>&1
  fi
  exit_code=$?
  printf 'EXECUTION: %s %s exit=%s seconds=%s output=%s\n' \
    "$video_key" "$scope" "$exit_code" "$(( $(date +%s) - started ))" "$output"
  tail -n 12 "$output.terminal.log"
  if [[ "$exit_code" -ne 0 ]]; then
    printf 'STOPPED: execution failed; later runs were not started.\n' >&2
    return 1
  fi
  if ! check_saved_run "$output" "$video_key" "$scope"; then
    printf 'STOPPED: saved-output gate failed; later runs were not started.\n' >&2
    return 1
  fi
}

run_check video5_imelda_0_part1 40 || exit 1
run_check video4 40 || exit 1
run_check video5_imelda_0_part1 full || exit 1
run_check video4 full || exit 1
printf '\nBATCH COMPLETE: all four runs and saved-output gates passed.\n'
printf 'Batch command elapsed: %s seconds\nWorkflow elapsed: not recorded\n' \
  "$(( $(date +%s) - batch_start ))"
