#!/usr/bin/env bash
#SBATCH --job-name=gcw-flow-cpu-gate
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00

set -euo pipefail

: "${GCW_PROJECT_DIR:?submit with GCW_PROJECT_DIR set to the staged project directory}"
: "${GCW_SOURCE_COMMIT:?submit with GCW_SOURCE_COMMIT set to the tested source commit}"
cd "$GCW_PROJECT_DIR"
PYTHON="$GCW_PROJECT_DIR/.venv/bin/python"
test -x "$PYTHON" || { printf 'Missing interpreter: %s\n' "$PYTHON" >&2; exit 1; }

printf 'job=%s node=%s source_commit=%s started=%s\n' \
  "$SLURM_JOB_ID" "$(hostname)" "$GCW_SOURCE_COMMIT" "$(date -u +%FT%TZ)"
"$PYTHON" -m pip check
"$PYTHON" -m pytest -q
"$PYTHON" -m ruff check .
"$PYTHON" -m mypy gcw
BASH_BIN="$(command -v bash)"
test -x "$BASH_BIN" || { printf 'Missing Bash interpreter\n' >&2; exit 1; }
"$BASH_BIN" -n scripts/slurm/flow_cpu_gate.sbatch scripts/slurm/flow_smoke.sbatch

SUCCESS_PATH="$GCW_PROJECT_DIR/cpu-gate-$SLURM_JOB_ID.ok"
set -o noclobber
printf 'source_commit=%s completed=%s\n' "$GCW_SOURCE_COMMIT" "$(date -u +%FT%TZ)" > "$SUCCESS_PATH"
printf 'success_marker=%s\n' "$SUCCESS_PATH"
