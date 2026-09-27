#!/usr/bin/env bash
# Prepared by Tahereh Fahi. Two independent fresh, bounded checks; no publication.
set -u

framework_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
workspace_dir="$(cd "${framework_dir}/.." && pwd)" || exit 1
cd "$workspace_dir" || exit 1
if [[ ! -x .venv/bin/python ]]; then
  printf 'Missing workspace Python environment: .venv/bin/python\n' >&2
  exit 1
fi
mkdir -p "${framework_dir}/runs"
pause_stamp="$(date +%Y%m%d_%H%M%S)_$$"
video5_output="${framework_dir}/runs/video5_part1_pause_all40_${pause_stamp}"
video4_output="${framework_dir}/runs/video4_pause_all40_${pause_stamp}"
pause_batch_start="$(date +%s)"

run_pause_check() {
  caffeinate -i env PYTHONPATH="${framework_dir}/src" \
    PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    .venv/bin/python -m vss_framework.cli run-video \
    --config "${framework_dir}/configs/$1.json" --output "$2" \
    --max-seconds 40 --stages all --no-resume >"$2.terminal.log" 2>&1
}

printf 'Running Video 5 part 1 and Video 4, first 40 seconds, in parallel.\n'
printf 'Logs:\n%s.terminal.log\n%s.terminal.log\n' "$video5_output" "$video4_output"
run_pause_check video5_imelda_0_part1 "$video5_output" &
video5_pid=$!
run_pause_check video4 "$video4_output" &
video4_pid=$!
wait "$video5_pid"
video5_exit=$?
wait "$video4_pid"
video4_exit=$?

printf '\nVideo5 exit=%s output=%s\n' "$video5_exit" "$video5_output"
tail -n 26 "${video5_output}.terminal.log"
printf '\nVideo4 exit=%s output=%s\n' "$video4_exit" "$video4_output"
tail -n 26 "${video4_output}.terminal.log"
printf '\nBatch command elapsed: %s seconds\nWorkflow elapsed: not recorded\n' "$(( $(date +%s) - pause_batch_start ))"
printf 'Send both exit/output lines and the two final summaries above.\n'
if [[ "$video5_exit" -ne 0 || "$video4_exit" -ne 0 ]]; then
  exit 1
fi
