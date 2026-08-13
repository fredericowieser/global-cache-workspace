#!/bin/bash
#SBATCH --job-name=martin_dev
#SBATCH --partition=agent-xlong,agent-xlong-15
#SBATCH --gres=gpu:2
#SBATCH --time=1-00:00:00

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOCAL_ENV="${SLURM_SUBMIT_DIR:-${REPO_ROOT}}/.env"

if [[ -f "${LOCAL_ENV}" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${LOCAL_ENV}"
  set +a
fi

DISCORD_WEBHOOK_URL="${TUNNEL_DISCORD_WEBHOOK_URL:-${DISCORD_WEBHOOK_URL:-}}"

json_escape() {
  local value="${1}"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  value="${value//$'\r'/}"
  value="${value//$'\n'/\\n}"
  printf '%s' "${value}"
}

notify_discord() {
  local message="${1}"

  if [[ -z "${DISCORD_WEBHOOK_URL}" ]]; then
    echo "Discord webhook URL is not configured; skipping notification" >&2
    return 0
  fi

  local escaped_message
  escaped_message="$(json_escape "${message}")"

  if ! curl -fsS \
    -H "Content-Type: application/json" \
    -d "{\"content\":\"${escaped_message}\",\"allowed_mentions\":{\"parse\":[]}}" \
    "${DISCORD_WEBHOOK_URL}" >/dev/null; then
    echo "Warning: failed to send Discord notification" >&2
  fi
}

CLI_PATH="$HOME/tmp/${USER}-vscode_cli"
echo "CLI_PATH: ${CLI_PATH}"

# Install the VS Code CLI command if it doesn't exist
if [[ ! -e ${CLI_PATH}/code ]]; then
  echo "Downloading and installing the VS Code CLI command"
  mkdir -p "${CLI_PATH}"
  pushd "${CLI_PATH}"
  # Process from: https://code.visualstudio.com/docs/remote/tunnels#_using-the-code-cli
  curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz
  # unpack the code binary file
  tar -xf vscode_cli.tar.gz
  # clean-up
  rm vscode_cli.tar.gz
  popd
fi

HOSTNAME_VALUE="$(hostname)"
STARTED_AT="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

notify_discord "VS Code tunnel starting
job: ${SLURM_JOB_ID:-no-slurm-job}
name: ${SLURM_JOB_NAME:-martin_dev}
node: ${SLURMD_NODENAME:-${HOSTNAME_VALUE}}
cwd: ${PWD}
time: ${STARTED_AT}"

# run the code tunnel command and accept the licence
srun ${CLI_PATH}/code tunnel --accept-server-license-terms
