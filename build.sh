#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

BUILD_TYPE="${BUILD_TYPE:-Release}"
GPU_ARCH="${GPU_ARCH:-native}"
GPU_PLATFORM="${GPU_PLATFORM:-${HIP_PLATFORM:-}}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
CLEAN="${CLEAN:-0}"
EXE="ppq.exe"

log() { printf '[build] %s\n' "$*"; }
die() { printf '[build] error: %s\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform)   GPU_PLATFORM="${2:?--platform needs a value}"; shift 2 ;;
        --gpu-arch)   GPU_ARCH="${2:?--gpu-arch needs a value}";     shift 2 ;;
        --build-type) BUILD_TYPE="${2:?--build-type needs a value}"; shift 2 ;;
        --jobs|-j)    JOBS="${2:?--jobs needs a value}";             shift 2 ;;
        --clean)      CLEAN=1; shift ;;
        *)            log "ignoring unrecognized argument: $1"; shift ;;
    esac
done

command -v cmake >/dev/null 2>&1 || die "cmake not found on PATH"
command -v make  >/dev/null 2>&1 || die "make not found on PATH"

if [[ -z "${GPU_PLATFORM}" ]]; then
    if command -v nvcc >/dev/null 2>&1; then
        GPU_PLATFORM=nvidia
    elif command -v hipconfig >/dev/null 2>&1; then
        GPU_PLATFORM=amd
    else
        die "no GPU toolchain found: need nvcc (NVIDIA) or hipconfig (AMD/ROCm), or set HIP_PLATFORM"
    fi
fi
[[ "${GPU_PLATFORM}" == "nvcc" ]] && GPU_PLATFORM=nvidia

case "${GPU_PLATFORM}" in
    nvidia) command -v nvcc >/dev/null 2>&1 || die "GPU_PLATFORM=nvidia but nvcc not found on PATH" ;;
    amd)    ;;
    *)      die "GPU_PLATFORM must be nvidia or amd, got '${GPU_PLATFORM}'" ;;
esac

BUILD_DIR="${BUILD_DIR:-build}"

log "config GPU_PLATFORM=${GPU_PLATFORM} GPU_ARCH=${GPU_ARCH} BUILD_TYPE=${BUILD_TYPE} JOBS=${JOBS}"

if [[ "${CLEAN}" == "1" ]]; then
    log "CLEAN=1, removing ${BUILD_DIR}/ and ${EXE}"
    rm -rf "${BUILD_DIR}" "${EXE}"
fi

build() {
    make GPU_PLATFORM="$1" GPU_ARCH="$2" BUILD_TYPE="${BUILD_TYPE}" BUILD_DIR="${BUILD_DIR}" JOBS="${JOBS}"
}

if ! build "${GPU_PLATFORM}" "${GPU_ARCH}"; then
    if [[ "${GPU_ARCH}" == "native" ]]; then
        case "${GPU_PLATFORM}" in
            nvidia) fallback="all-major" ;;
            amd)    fallback="gfx1151" ;;
        esac
        log "native architecture detection failed, retrying with GPU_ARCH=${fallback}"
        rm -rf "${BUILD_DIR}"
        build "${GPU_PLATFORM}" "${fallback}" || die "build failed"
    else
        die "build failed"
    fi
fi

[[ -x "./${EXE}" ]] || die "build reported success but ./${EXE} is missing"

log "ok: $(ls -lh "./${EXE}" | awk '{print $5}') -> ./${EXE}"
