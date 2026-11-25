# Microbenchmarks (Google Benchmark). bench_host.exe needs no GPU toolchain; bench_gpu.exe
# times each kernel stage with HIP events. Both take --ppq_size=smoke|quick|full.
#
# The smoke runs are CTest tests (label bench): every benchmark runs once on a small input,
# which checks that they work, not how fast anything is.

function(ppq_add_bench name)
    cmake_parse_arguments(PARSE_ARGV 1 arg "" "" "SOURCES;LIBS")
    add_executable(${name} ${arg_SOURCES})
    target_link_libraries(${name} PRIVATE ${arg_LIBS} benchmark::benchmark ppq_warnings)
    set_target_properties(${name} PROPERTIES SUFFIX ".exe")
    add_test(NAME ${name}_smoke
        COMMAND ${name} --ppq_size=smoke --benchmark_min_time=1x)
endfunction()

ppq_add_bench(bench_host SOURCES bench_host.cpp LIBS ppq_cpuref)
set_tests_properties(bench_host_smoke PROPERTIES LABELS bench)

if(NOT PPQ_GPU_ENABLED)
    return()
endif()

ppq_gpu_sources(bench_gpu.hip)
ppq_add_bench(bench_gpu SOURCES bench_gpu.hip LIBS ppq_core ppq_cpuref)
# exits 77 when no device is visible
set_tests_properties(bench_gpu_smoke PROPERTIES LABELS "bench;gpu" SKIP_RETURN_CODE 77)
