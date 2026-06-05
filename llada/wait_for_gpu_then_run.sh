#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <npu> <command> [args...]"
  exit 1
fi

NPU="$1"
shift

if command -v npu-smi >/dev/null 2>&1; then
  npu-smi info -i "${NPU}" >/dev/null 2>&1 || npu-smi info >/dev/null 2>&1 || true
fi

export ASCEND_RT_VISIBLE_DEVICES="${NPU}"
export NPU_VISIBLE_DEVICES="${NPU}"

echo "$(date '+%F %T') NPU ${NPU} is selected; starting: $*"
exec "$@"
