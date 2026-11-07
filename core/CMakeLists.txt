add_library(ppq_host STATIC
    src/shake_vectors.cpp
    src/mldsa_cpu.cpp
    src/workload.cpp
    src/kv_vectors.cpp
    src/acvp.cpp
)
find_package(Threads REQUIRED)
target_include_directories(ppq_host PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ppq_host PUBLIC Threads::Threads PRIVATE ppq_warnings)

if(NOT PPQ_GPU_ENABLED)
    return()
endif()

set(PPQ_CORE_GPU_SOURCES
    src/device.hip
    src/shake_gpu.hip
    src/hash_kernels.hip
)

ppq_gpu_sources(${PPQ_CORE_GPU_SOURCES})

add_library(ppq_core STATIC ${PPQ_CORE_GPU_SOURCES})
target_include_directories(ppq_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ppq_core PUBLIC ppq_host ppq_gpu_runtime PRIVATE ppq_warnings)

ppq_gpu_resource_usage(ppq_core)
