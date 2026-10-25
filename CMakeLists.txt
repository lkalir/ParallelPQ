cmake_minimum_required(VERSION 3.24)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

project(ParallelPQ
    VERSION 0.1.0
    DESCRIPTION "Batched GPU ML-DSA (FIPS 204) signature verification"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(PPQ_GPU_ENABLED "Build the GPU targets; OFF builds only the host libraries and tests" ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
if(PPQ_GPU_ENABLED)
    include(GpuPlatform)
endif()

add_library(ppq_warnings INTERFACE)
target_compile_options(ppq_warnings INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
    $<$<COMPILE_LANGUAGE:HIP>:-Wall -Wextra>)

enable_testing()
add_subdirectory(core)
add_subdirectory(tests)

if(NOT PPQ_GPU_ENABLED)
    return()
endif()

function(ppq_add_exe target)
    ppq_gpu_sources(${ARGN})
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} PRIVATE ppq_core ppq_warnings)
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/apps)
    target_compile_definitions(${target} PRIVATE PPQ_VERSION="${PROJECT_VERSION}")
    set_target_properties(${target} PROPERTIES SUFFIX ".exe")
endfunction()

ppq_add_exe(ppq apps/ppq.hip)
ppq_add_exe(probe tools/probe.hip)
