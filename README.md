# ParallelPQ (ppq)

Batched GPU verification of ML-DSA signatures
([FIPS 204](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.204.pdf)),
created for JHU EP EN.605.617.

ParallelPQ is a standalone application written in HIP for compatibility
with NVIDIA and AMD GPUs. It either generates keys and signatures at
run time or loads a pre-generated set from a `.ppq` file, verifies them
as a batch, and checks every result against the expected verdict.

## Current status

The project currently contains a complete CPU reference implementation
of the application based on
[pq-crystals/dilithium](https://github.com/pq-crystals/dilithium)
that has been verified against standard NIST test vectors. It can
also optionally use OpenSSL 3.5+ for CPU validation. The CPU reference
is capable of generating `.ppq` files, validating them, and passing
known answer tests.

The GPU implementation is a work in progress. It contains a complete
and validated

- SHAKE implementation (from a past assignment)
- Number theoretic transform (NTT)
- Montgomery multiplication
- Public key and signature decoding
- tr and mu computation

The remaining GPU work includes

- ExpandA (recreates the public matrix from the public seed)
- Key caching (cache key-derived items)
- SampleInBall (converts commitment hash into challenge polynomial)
- Core arithmetic using NTT and inverse NTT
- UseHint (correcting approximations and decomposing)
- Computing and validating the final hash

Benchmarks will also be created to characterize performance and guide
further optimization, time permitting.

## Build

### Requirements

- CMake 3.24+
- A C++20 compiler and a C compiler. GCC 13+ and Clang have both been
used.
- A CUDA toolkit for NVIDIA, or ROCm for AMD. The build system
downloads the HIP headers.
- Optional: OpenSSL 3.5+
- Python 3, only to regenerate test vectors with the scripts in
`tools/`.

A Nix flake provides a development shell with all of the above:

```sh
nix develop
```

### Building

Builds can either be done using `make`, the provided `build.sh` script,
or using `cmake` directly.

#### With make

The Makefile wraps CMake. GPU builds go to `build/` and CPU-only builds
to `build-host/`. Targets that produce `ppq.exe` or `probe.exe` copy it
to the repository root.

| Command | What it does |
|---|---|
| `make` | GPU build of `ppq.exe` |
| `make cpu` | CPU-only `ppq.exe` (no GPU toolchain needed) |
| `make test` | Builds everything (GPU) and runs all tests |
| `make test-host` | CPU-only build and its tests |
| `make probe` | Builds `probe.exe`, a GPU smoke test |
| `make run RUN_ARGS="kat"` | Builds and runs `ppq.exe` with those arguments (default `info`) |
| `make stage-vectors` | Regenerates `tests/vectors/stages` from pq-crystals |
| `make clean` / `make distclean` | Cleans the build / removes both build directories and copied executables |

Make variables:

| Variable | Meaning | Default |
|---|---|---|
| `GPU_PLATFORM` | `nvidia` or `amd` | `$HIP_PLATFORM`, else `nvidia` if `nvcc` is on PATH, else `amd` if `hipconfig` is |
| `GPU_ARCH` | `native`, an NVIDIA SM such as `86`, or an AMD target such as `gfx1100` | `native` |
| `BUILD_TYPE` | CMake build type | `Release` |
| `BUILD_DIR`, `HOST_BUILD_DIR` | Build directories | `build`, `build-host` |
| `JOBS` | Parallel build jobs | `nproc` |

If `build/` was configured for the other GPU platform, `make` removes
it and reconfigures.

#### With build.sh

```sh
./build.sh [--platform nvidia|amd] [--gpu-arch ARCH] [--build-type TYPE] [-j N] [--clean]
```

It detects the platform like `make`, then builds `ppq.exe`. If
`GPU_ARCH=native` fails (for example, no GPU visible at build time), it
retries with `all-major` on NVIDIA or `gfx1151` on AMD.

#### With CMake directly

```sh
cmake -B build -DPPQ_GPU_PLATFORM=nvidia -DPPQ_GPU_ARCH=86
cmake --build build -j
```

| Option | Meaning | Default |
|---|---|---|
| `PPQ_GPU_ENABLED` | Build the GPU targets; `OFF` builds the host libraries, `ppq.exe` and host tests only | `ON` |
| `PPQ_GPU_PLATFORM` | `nvidia` or `amd` | `$HIP_PLATFORM` |
| `PPQ_GPU_ARCH` | GPU architecture(s) | `native` |
| `PPQ_GPU_FETCH_HIP_HEADERS` | On NVIDIA, download the HIP headers if none are found | `ON` |
| `PPQ_GPU_HIP_HEADERS_TAG` | ROCm tag for those headers | `rocm-7.2.4` |
| `PPQ_GPU_RESOURCE_USAGE` | Print each kernel's registers, spills and scratch while compiling | `OFF` |
| `PPQ_CPUREF_AVX2` | Build the pq-crystals AVX2 implementation | `ON` on x86-64 |
| `PPQ_CPU_NATIVE` | Compile the CPU reference with `-march=native` | `ON` |
| `PPQ_WITH_OPENSSL` | Use OpenSSL 3.5+ when found | `ON` |
| `PPQ_DILITHIUM_COMMIT` | pq-crystals commit to fetch | `d35ba3fe…` |

Executables are written to `build/bin/` (or `build-host/bin/`).

## Running

```text
usage: ppq.exe <command> [options]

  info        CPU implementations, threads and GPUs available
  generate    write a workload of keys and signatures to a .ppq file
  verify      verify a generated or loaded workload and check the results
  kat         verify the NIST ACVP vectors
```

`ppq.exe --help` lists every option.

### Examples

```sh
./ppq.exe info

# 2,048 ML-DSA-65 signatures over 16 keys, half of them tampered in every way,
# with the reason for each failure, timed over 3 runs
./ppq.exe verify -p 65 -n 2048 -k 16 --pass-rate 0.5 --tamper mixed --verdicts -r 3

# Write a workload to a file, then verify it
./ppq.exe generate -p 87 -n 4096 -k 1 -o same-key.ppq
./ppq.exe verify -i same-key.ppq --format json

# All 135 NIST ACVP sigVer vectors (external, internal and mu interfaces)
./ppq.exe kat
```

Output from the `verify` example (timings depend on the machine):

```text
ParallelPQ 0.1.0 verify
workload  ML-DSA-65, external, 2,048 signatures, 16 keys, 1,024 expected to pass
          generated: seed 1, pq-crystals AVX2, 26.9 ms
backend   CPU: pq-crystals ref stages (with verdicts), 24 threads
time      22.8 ms best, 23.6 ms median, 23.9 ms mean over 3 runs
rate      89,782 signatures/s
result    1,024 valid, 1,024 invalid
verdicts  pass 1024, bad-commitment 512, z-norm 256, bad-hint 256
check     all 2,048 agree with the expected verdicts
```

### run.sh

`./run.sh` builds `ppq.exe` if needed, then runs a default suite:
`info`, `kat`, each parameter set, a same-key workload, and a 50% pass
rate with every tamper mode. It saves each run's output under
`results/<gpu-name>/`. The commands run can be overridden via the
command line.

- `RUNS="verify -p 44 -n 8192;kat" ./run.sh` replaces the suite.
- `./run.sh verify -p 87 -n 1024` runs a single command.
- `./run.sh --test` runs `make test`.

## Testing

```sh
make test-host   # CPU only
make test        # GPU build and tests, plus CPU tests
```

Tests include both unit tests and known answer tests against test
vectors
([CAVP](https://csrc.nist.gov/Projects/Cryptographic-Algorithm-Validation-Program)
and [ACVP](https://pages.nist.gov/ACVP/) suites).

## Source directory

```text
.
|-- CMakeLists.txt            (see Building)
|-- Makefile                  (see Building)
|-- build.sh, run.sh          (see Building, Running)
|-- flake.nix, flake.lock     Nix development shell
|-- cmake/                    GPU/HIP autodetection
|-- apps/                     ppq.exe
|-- core/                     ParallelPQ's host and GPU code
|   |-- CMakeLists.txt        ppq_host (plain C++) and ppq_core (GPU) libraries
|   |-- include/ppq/          ppq headers
|   |-- src/                  Implementations (.cpp host, .hip device)
|-- cpuref/                   CPU reference implementation
|   |-- CMakeLists.txt        Fetches pq-crystals, builds ref and AVX2
|   |-- include/ppq/          CPU reference implementation headers
|   |-- src/                  pq-crystals and OpenSSL wrappers
|-- tests/                    Test code
|-- tools/                    Helper utilities for vector generation
```
