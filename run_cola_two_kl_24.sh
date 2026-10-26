#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
GPU_ID="${GPU_ID:-0}"
if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "GPU_ID must identify one physical GPU" >&2
    exit 2
fi
# Resolve Python now: nohup inherits the active Conda environment.
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
LOG_DIR="$PROJECT_ROOT/logs/cola_two_kl_24"
mkdir -p "$LOG_DIR"
exec 9>"$LOG_DIR/queue.lock"
if ! flock -n 9; then
    echo "Another cola_two_kl_24 queue is running" >&2
    exit 75
fi
export GPU_ID
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TZ=Asia/Shanghai
if [[ "${1:-}" == "--dry-run" ]]; then
    exec "$PYTHON_BIN" -u cola_two_kl_queue.py --dry-run
fi
if (( $# )); then
    echo "Usage: GPU_ID=0 $0 [--dry-run]" >&2
    exit 2
fi
exec > >(tee -a "$LOG_DIR/queue.log") 2>&1
printf 'Queue start: PID=%s GPU=%s time=%s Python=%s log=%s\n' "$$" "$GPU_ID" "$(date -Is)" "$PYTHON_BIN" "$LOG_DIR/queue.log"
exec "$PYTHON_BIN" -u cola_two_kl_queue.py
