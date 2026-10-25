set(PPQ_TEST_VECTORS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vectors")

function(ppq_add_host_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE ppq_host ppq_warnings)
    target_compile_definitions(${name} PRIVATE TEST_VECTORS_DIR="${PPQ_TEST_VECTORS_DIR}")
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES LABELS host)
endfunction()

ppq_add_host_test(test_launch_host)
ppq_add_host_test(test_shake_host)
ppq_add_host_test(test_mldsa_host)

if(NOT PPQ_GPU_ENABLED)
    return()
endif()

function(ppq_add_gpu_test name)
    ppq_gpu_sources(${name}.hip)
    add_executable(${name} ${name}.hip)
    target_link_libraries(${name} PRIVATE ppq_core ppq_warnings)
    target_compile_definitions(${name} PRIVATE TEST_VECTORS_DIR="${PPQ_TEST_VECTORS_DIR}")
    add_test(NAME ${name} COMMAND ${name})
    # gpu_test::run_all() exits 77 when no device is visible
    set_tests_properties(${name} PROPERTIES LABELS gpu SKIP_RETURN_CODE 77)
endfunction()

ppq_add_gpu_test(test_runtime)
ppq_add_gpu_test(test_shake_gpu)
ppq_add_gpu_test(test_mldsa_hash)
