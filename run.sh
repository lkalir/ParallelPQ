#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

EXE="./ppq.exe"
# Runs separated by ";"; later phases add verify and benchmark runs here
DEFAULT_RUNS="info"
RUNS="${RUNS:-${DEFAULT_RUNS}}"

log() { printf '[run] %s\n' "$*"; }
die() { printf '[run] error: %s\n' "$*" >&2; exit 1; }

if [[ "${1:-}" == "--test" ]]; then
    exec make test
fi

if [[ ! -x "${EXE}" ]]; then
    log "${EXE} not found, building first"
    ./build.sh
fi

[[ -x "${EXE}" ]] || die "${EXE} is missing or not executable"

gpu_name() {
    if command -v nvidia-smi >/dev/null 2>&1 &&
        nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name --format=csv,noheader | head -1
        return
    fi

    if command -v rocminfo >/dev/null 2>&1; then
        rocminfo 2>/dev/null |
            awk -F: '/^ *Name: *gfx/ {gpu=1} gpu && /Marketing Name/ {sub(/^ +/, "", $2); print $2; exit}'
        return
    fi
}

name="$(gpu_name || true)"
if [[ -n "${name}" ]]; then
    log "gpu: ${name}"
    slug="$(printf '%s' "${name}" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '-' | sed 's/^-//; s/-$//')"
else
    log "warning: could not identify the GPU"
    slug="${HIP_PLATFORM:-unknown-gpu}"
fi

RESULTS_DIR="${RESULTS_DIR:-results/${slug}}"
mkdir -p "${RESULTS_DIR}"

status=0

run_logged() {
    local out="$1"
    shift
    log "running: ${EXE} $*"
    set +e
    "${EXE}" "$@" 2>&1 | tee "${out}"
    local rc=${PIPESTATUS[0]}
    set -e
    if [[ ${rc} -ne 0 ]]; then
        log "${EXE} exited with status ${rc}"
        status=${rc}
    fi
}

if [[ $# -gt 0 ]]; then
    run_logged "${RESULTS_DIR}/last-run.txt" "$@"
    exit "${status}"
fi

IFS=';' read -ra runs <<< "${RUNS}"
for r in "${runs[@]}"; do
    read -ra args <<< "${r}"
    file="$(printf '%s' "${args[*]}" | tr -cs 'a-zA-Z0-9' '-' | sed 's/^-//; s/-$//')"
    printf '\n'
    run_logged "${RESULTS_DIR}/${file}.txt" "${args[@]}"
done

log "results in ${RESULTS_DIR}/"
exit "${status}"
