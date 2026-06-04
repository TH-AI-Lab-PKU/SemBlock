#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 <gpu> <complete_path> <expected_lines|exists> <queue_log> <command> [args...]"
  exit 1
fi

GPU="$1"
COMPLETE_PATH="$2"
EXPECTED="$3"
QUEUE_LOG="$4"
shift 4

MAX_USED_MIB="${MAX_USED_MIB:-60000}"
SLEEP_SECONDS="${SLEEP_SECONDS:-300}"

is_done() {
  if [[ "${EXPECTED}" == "exists" ]]; then
    [[ -f "${COMPLETE_PATH}" ]]
    return
  fi
  if [[ ! -f "${COMPLETE_PATH}" ]]; then
    return 1
  fi
  local lines
  lines="$(wc -l < "${COMPLETE_PATH}")"
  [[ "${lines}" -ge "${EXPECTED}" ]]
}

mkdir -p "$(dirname "${QUEUE_LOG}")"

while ! is_done; do
  while true; do
    used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU}" | tr -d '[:space:]')"
    if [[ "${used_mib}" =~ ^[0-9]+$ && "${used_mib}" -le "${MAX_USED_MIB}" ]]; then
      break
    fi
    echo "$(date '+%F %T') waiting for GPU ${GPU}: used ${used_mib:-unknown} MiB > ${MAX_USED_MIB} MiB" | tee -a "${QUEUE_LOG}"
    sleep "${SLEEP_SECONDS}"
  done

  echo "$(date '+%F %T') starting: $*" | tee -a "${QUEUE_LOG}"
  set +e
  "$@" >> "${QUEUE_LOG}" 2>&1
  status="$?"
  set -e
  if is_done; then
    break
  fi
  echo "$(date '+%F %T') command exited with ${status}; retrying after ${SLEEP_SECONDS}s" | tee -a "${QUEUE_LOG}"
  sleep "${SLEEP_SECONDS}"
done

echo "$(date '+%F %T') complete: ${COMPLETE_PATH}" | tee -a "${QUEUE_LOG}"
