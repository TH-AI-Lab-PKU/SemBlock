#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 <npu> <complete_path> <expected_lines|exists> <queue_log> <command> [args...]"
  exit 1
fi

NPU="$1"
COMPLETE_PATH="$2"
EXPECTED="$3"
QUEUE_LOG="$4"
shift 4

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

export ASCEND_RT_VISIBLE_DEVICES="${NPU}"
export NPU_VISIBLE_DEVICES="${NPU}"

while ! is_done; do
  if command -v npu-smi >/dev/null 2>&1; then
    npu-smi info -i "${NPU}" >> "${QUEUE_LOG}" 2>&1 || npu-smi info >> "${QUEUE_LOG}" 2>&1 || true
  fi

  echo "$(date '+%F %T') starting on NPU ${NPU}: $*" | tee -a "${QUEUE_LOG}"
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
