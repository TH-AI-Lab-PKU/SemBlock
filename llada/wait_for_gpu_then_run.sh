#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <gpu> <command> [args...]"
  exit 1
fi

GPU="$1"
shift
MAX_USED_MIB="${MAX_USED_MIB:-20000}"

while true; do
  USED_MIB="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU}" | tr -d '[:space:]')"
  if [[ "${USED_MIB}" =~ ^[0-9]+$ && "${USED_MIB}" -le "${MAX_USED_MIB}" ]]; then
    break
  fi
  echo "$(date '+%F %T') waiting for GPU ${GPU}: used ${USED_MIB:-unknown} MiB > ${MAX_USED_MIB} MiB"
  sleep 300
done

echo "$(date '+%F %T') GPU ${GPU} is available; starting: $*"
exec "$@"
