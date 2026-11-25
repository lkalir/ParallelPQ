cmake_minimum_required(VERSION 3.24)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

project(ParallelPQ
    VERSION 0.1.0
    DESCRIPTION "Batched GPU ML-DSA (FIPS 204) signature verification"
    LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(PPQ_GPU_ENABLED "Build the GPU targets; OFF builds only the host libraries and tests" ON)

# CPU reference (cpuref/)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    set(_ppq_x86_64 ON)
else()
    set(_ppq_x86_64 OFF)
endif()
option(PPQ_CPUREF_AVX2 "Build the pq-crystals AVX2 implementation (x86-64)" ${_ppq_x86_64})
option(PPQ_CPU_NATIVE "Compile the CPU reference with -march=native" ON)
option(PPQ_WITH_OPENSSL "Use OpenSSL 3.5+ as an extra CPU implementation when found" ON)
option(PPQ_BUILD_BENCHMARKS "Build the Google Benchmark microbenchmarks (bench/)" ON)

if(PPQ_CPUREF_AVX2)
    enable_language(ASM)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
if(PPQ_GPU_ENABLED)
    include(GpuPlatform)
endif()

add_library(ppq_warnings INTERFACE)
target_compile_options(ppq_warnings INTERFACE
    $<$<COMPILE_LANGUAGE:C,CXX>:-Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
    $<$<COMPILE_LANGUAGE:HIP>:-Wall -Wextra>)

# Command line and JSON for ppq.exe: an installed package (e.g. from the Nix dev shell) is
# used when found, otherwise the pinned release is fetched
include(FetchContent)
FetchContent_Declare(CLI11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG        v2.7.2
    GIT_SHALLOW    TRUE
    FIND_PACKAGE_ARGS 2.4 CONFIG)
set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.12.0
    GIT_SHALLOW    TRUE
    FIND_PACKAGE_ARGS 3.11 CONFIG)
FetchContent_MakeAvailable(CLI11 nlohmann_json)

if(PPQ_BUILD_BENCHMARKS)
    FetchContent_Declare(benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG        v1.9.4
        GIT_SHALLOW    TRUE
        FIND_PACKAGE_ARGS 1.8 CONFIG)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(benchmark)
endif()

enable_testing()
add_subdirectory(core)
add_subdirectory(cpuref)

# ppq.exe: the application layer is host C++ on top of the CPU reference, so it builds
# and runs without a GPU toolchain; GPU builds add the device query (and later the GPU
# verifier) below.
add_library(ppq_app STATIC
    apps/options.cpp
    apps/commands.cpp
)
target_include_directories(ppq_app PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/apps)
target_link_libraries(ppq_app
    PUBLIC ppq_cpuref nlohmann_json::nlohmann_json
    PRIVATE CLI11::CLI11 ppq_warnings)
target_compile_definitions(ppq_app PRIVATE
    PPQ_VERSION="${PROJECT_VERSION}"
    PPQ_VECTORS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors")

add_executable(ppq apps/main.cpp)
target_link_libraries(ppq PRIVATE ppq_app ppq_warnings)
set_target_properties(ppq PROPERTIES SUFFIX ".exe")

# Host tool that regenerates tests/vectors/stages (make stage-vectors)
add_executable(stage_vectors tools/stage_vectors.cpp)
target_link_libraries(stage_vectors PRIVATE ppq_cpuref ppq_warnings)
set_target_properties(stage_vectors PROPERTIES SUFFIX ".exe")

add_subdirectory(tests)
if(PPQ_BUILD_BENCHMARKS)
    add_subdirectory(bench)
endif()

if(NOT PPQ_GPU_ENABLED)
    target_compile_definitions(ppq PRIVATE PPQ_WITH_GPU=0)
    return()
endif()

ppq_gpu_sources(apps/gpu_query.hip)
target_sources(ppq PRIVATE apps/gpu_query.hip)
target_link_libraries(ppq PRIVATE ppq_core)
target_compile_definitions(ppq PRIVATE PPQ_WITH_GPU=1)

ppq_gpu_sources(tools/probe.hip)
add_executable(probe tools/probe.hip)
target_link_libraries(probe PRIVATE ppq_core ppq_warnings)
set_target_properties(probe PROPERTIES SUFFIX ".exe")
