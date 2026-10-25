include_guard(GLOBAL)

set(PPQ_GPU_PLATFORM "" CACHE STRING "GPU platform: nvidia or amd (default: $HIP_PLATFORM)")
set(PPQ_GPU_ARCH "native" CACHE STRING "GPU architecture(s): native, an NVIDIA SM or an AMD target")
option(PPQ_GPU_FETCH_HIP_HEADERS
       "On NVIDIA, download the header-only HIP layer when no local copy with nvidia_detail is found" ON)
set(PPQ_GPU_HIP_HEADERS_TAG "rocm-7.2.4" CACHE STRING "ROCm release tag used when downloading HIP headers")
option(PPQ_GPU_RESOURCE_USAGE "Print per-kernel register, spill and scratch usage while compiling" OFF)

if(NOT PPQ_GPU_PLATFORM)
    set(_gpu_platform "$ENV{HIP_PLATFORM}")

    if(NOT _gpu_platform)
        find_program(_gpu_nvcc nvcc)
        find_program(_gpu_hipconfig hipconfig)

        if(_gpu_nvcc)
            set(_gpu_platform nvidia)
        elseif(_gpu_hipconfig)
            set(_gpu_platform amd)
        endif()
    endif()

    # hipcc spells these differently
    if(_gpu_platform STREQUAL "nvcc")
        set(_gpu_platform nvidia)
    elseif(_gpu_platform MATCHES "^(hcc|clang)$")
        set(_gpu_platform amd)
    endif()

    set(PPQ_GPU_PLATFORM "${_gpu_platform}" CACHE STRING "GPU platform: nvidia or amd" FORCE)
endif()

if(NOT PPQ_GPU_PLATFORM MATCHES "^(nvidia|amd)$")
    message(FATAL_ERROR
        "Could not determine the GPU platform (got '${PPQ_GPU_PLATFORM}'). Set HIP_PLATFORM=nvidia|amd")
endif()

message(STATUS "GPU platform: ${PPQ_GPU_PLATFORM}")

if(PPQ_GPU_PLATFORM STREQUAL "nvidia")
    set(PPQ_GPU_LANG CUDA)

    set(CMAKE_CUDA_ARCHITECTURES "${PPQ_GPU_ARCH}")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)

    # Fix for clangd
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)

    # Faster cudart search on nixos
    get_filename_component(_gpu_nvcc_root "${CMAKE_CUDA_COMPILER}" DIRECTORY)
    get_filename_component(_gpu_nvcc_root "${_gpu_nvcc_root}" DIRECTORY)
    set(_gpu_cuda_roots ${CUDAToolkit_ROOT} $ENV{CUDAToolkit_ROOT} $ENV{CUDA_PATH} ${_gpu_nvcc_root})

    find_library(PPQ_GPU_CUDART_LIBRARY
        NAMES cudart
        HINTS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} ${_gpu_cuda_roots}
        PATH_SUFFIXES lib64 lib lib/x64
        NO_DEFAULT_PATH)

    if(PPQ_GPU_CUDART_LIBRARY)
        get_filename_component(_gpu_cudart_dir "${PPQ_GPU_CUDART_LIBRARY}" DIRECTORY)
    endif()

    find_path(PPQ_GPU_CUDART_INCLUDE_DIR
        NAMES cuda_runtime.h
        HINTS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} "${_gpu_cudart_dir}/../include"
              ${_gpu_cuda_roots}
        PATH_SUFFIXES include
        NO_DEFAULT_PATH)

    add_library(ppq_gpu_runtime INTERFACE)

    if(PPQ_GPU_CUDART_LIBRARY AND PPQ_GPU_CUDART_INCLUDE_DIR)
        message(STATUS "CUDA runtime: ${PPQ_GPU_CUDART_LIBRARY}")
        target_link_libraries(ppq_gpu_runtime INTERFACE "${PPQ_GPU_CUDART_LIBRARY}")
        target_include_directories(ppq_gpu_runtime SYSTEM INTERFACE "${PPQ_GPU_CUDART_INCLUDE_DIR}")
    else()
        message(STATUS "cudart not found next to nvcc, falling back to FindCUDAToolkit")
        find_package(CUDAToolkit REQUIRED)
        target_link_libraries(ppq_gpu_runtime INTERFACE CUDA::cudart)
    endif()

    find_path(PPQ_GPU_HIP_INCLUDE_DIR
        NAMES hip/nvidia_detail/nvidia_hip_runtime.h
        HINTS ENV HIP_PATH ENV ROCM_PATH /opt/rocm
        PATH_SUFFIXES include
        NO_DEFAULT_PATH)

    if(NOT PPQ_GPU_HIP_INCLUDE_DIR)
        if(NOT PPQ_GPU_FETCH_HIP_HEADERS)
            message(FATAL_ERROR
                "No HIP headers with hip/nvidia_detail found (looked in $HIP_PATH, $ROCM_PATH, /opt/rocm)"
            )
        endif()

        message(STATUS "Fetching header-only HIP for NVIDIA (${PPQ_GPU_HIP_HEADERS_TAG})")
        include(FetchContent)

        FetchContent_Declare(gpu_hip_api
            GIT_REPOSITORY https://github.com/ROCm/hip.git
            GIT_TAG        ${PPQ_GPU_HIP_HEADERS_TAG}
            GIT_SHALLOW    TRUE
            SOURCE_SUBDIR  _headers_only)
        FetchContent_Declare(gpu_hip_nvidia
            GIT_REPOSITORY https://github.com/ROCm/hipother.git
            GIT_TAG        ${PPQ_GPU_HIP_HEADERS_TAG}
            GIT_SHALLOW    TRUE
            SOURCE_SUBDIR  _headers_only)
        FetchContent_Declare(gpu_hip_clr
            GIT_REPOSITORY https://github.com/ROCm/clr.git
            GIT_TAG        ${PPQ_GPU_HIP_HEADERS_TAG}
            GIT_SHALLOW    TRUE
            SOURCE_SUBDIR  _headers_only)
        FetchContent_MakeAvailable(gpu_hip_api gpu_hip_nvidia gpu_hip_clr)

        set(_gpu_hip_dir "${CMAKE_BINARY_DIR}/hip-nvidia/include")
        file(COPY "${gpu_hip_api_SOURCE_DIR}/include/hip" DESTINATION "${_gpu_hip_dir}")
        file(COPY "${gpu_hip_nvidia_SOURCE_DIR}/hipnv/include/hip/nvidia_detail"
             DESTINATION "${_gpu_hip_dir}/hip")
        file(COPY "${gpu_hip_clr_SOURCE_DIR}/hipamd/include/hip/amd_detail"
             DESTINATION "${_gpu_hip_dir}/hip")

        # hip_version.h is generated by the ROCm build, build it ourselves
        string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\.?([0-9]*)" _gpu_ver "${PPQ_GPU_HIP_HEADERS_TAG}")
        set(_gpu_major "${CMAKE_MATCH_1}")
        set(_gpu_minor "${CMAKE_MATCH_2}")
        set(_gpu_patch "${CMAKE_MATCH_3}")
        if(NOT _gpu_patch)
            set(_gpu_patch 0)
        endif()
        file(WRITE "${_gpu_hip_dir}/hip/hip_version.h"
            "// Generated by cmake/GpuPlatform.cmake\n"
            "#pragma once\n"
            "#define HIP_VERSION_MAJOR ${_gpu_major}\n"
            "#define HIP_VERSION_MINOR ${_gpu_minor}\n"
            "#define HIP_VERSION_PATCH ${_gpu_patch}\n"
            "#define HIP_VERSION_GITHASH \"${PPQ_GPU_HIP_HEADERS_TAG}\"\n"
            "#define HIP_VERSION_BUILD_ID 0\n"
            "#define HIP_VERSION_BUILD_NAME \"\"\n"
            "#define HIP_VERSION (HIP_VERSION_MAJOR * 10000000 + HIP_VERSION_MINOR * 100000 + HIP_VERSION_PATCH)\n")

        set(PPQ_GPU_HIP_INCLUDE_DIR "${_gpu_hip_dir}" CACHE PATH "HIP headers with nvidia_detail" FORCE)
    endif()

    message(STATUS "HIP headers: ${PPQ_GPU_HIP_INCLUDE_DIR}")
    target_include_directories(ppq_gpu_runtime SYSTEM INTERFACE "${PPQ_GPU_HIP_INCLUDE_DIR}")
    target_compile_definitions(ppq_gpu_runtime INTERFACE __HIP_PLATFORM_NVIDIA__)
else()
    set(PPQ_GPU_LANG HIP)

    set(_gpu_arch "${PPQ_GPU_ARCH}")

    # CMake has no "native" for HIP, so probe it ourselves
    if(_gpu_arch STREQUAL "native")
        set(_gpu_arch "")
        execute_process(COMMAND hipconfig --hipclangpath
            OUTPUT_VARIABLE _gpu_clang_dir OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        find_program(_gpu_amdgpu_arch NAMES amdgpu-arch rocm_agent_enumerator
            HINTS "${_gpu_clang_dir}" ENV ROCM_PATH PATH_SUFFIXES bin llvm/bin)

        if(_gpu_amdgpu_arch)
            execute_process(COMMAND "${_gpu_amdgpu_arch}"
                OUTPUT_VARIABLE _gpu_agents ERROR_QUIET)
            string(REGEX MATCHALL "gfx[0-9a-f]+" _gpu_agents "${_gpu_agents}")
            list(REMOVE_ITEM _gpu_agents gfx000)
            list(REMOVE_DUPLICATES _gpu_agents)
            set(_gpu_arch "${_gpu_agents}")
        endif()

        if(NOT _gpu_arch)
            message(FATAL_ERROR
                "PPQ_GPU_ARCH=native but no AMD GPU was detected, name the target instead"
            )
        endif()
    endif()

    set(CMAKE_HIP_ARCHITECTURES "${_gpu_arch}")
    enable_language(HIP)
    set(CMAKE_HIP_STANDARD 20)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    set(CMAKE_HIP_EXTENSIONS OFF)
    message(STATUS "HIP architectures: ${CMAKE_HIP_ARCHITECTURES}")
    add_library(ppq_gpu_runtime INTERFACE)
endif()

# Compile the given .hip files with whichever compiler drives this platform.
function(ppq_gpu_sources)
    set_source_files_properties(${ARGN} PROPERTIES LANGUAGE ${PPQ_GPU_LANG})
endfunction()

function(ppq_gpu_resource_usage target)
    if(NOT PPQ_GPU_RESOURCE_USAGE)
        return()
    endif()

    if(PPQ_GPU_PLATFORM STREQUAL "nvidia")
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)
    else()
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:HIP>:-Rpass-analysis=kernel-resource-usage>)
    endif()
endfunction()
