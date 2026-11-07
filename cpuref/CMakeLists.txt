# ppq_cpuref: third-party CPU ML-DSA used to generate workloads and as the baseline.
#
# pq-crystals is fetched at a pinned commit. To build offline, point
# FETCHCONTENT_SOURCE_DIR_PQCRYSTALS_DILITHIUM at a local checkout of that commit.

include(FetchContent)

# 2026-06-03: includes the AVX2 use_hint fix and the stricter z norm check
set(PPQ_DILITHIUM_COMMIT d35ba3fe5449bee3e6d43e1f296c3ca818bd36be
    CACHE STRING "pq-crystals/dilithium commit for the CPU reference")

FetchContent_Declare(pqcrystals_dilithium
    GIT_REPOSITORY https://github.com/pq-crystals/dilithium.git
    GIT_TAG        ${PPQ_DILITHIUM_COMMIT})
FetchContent_MakeAvailable(pqcrystals_dilithium)

set(_pqc "${pqcrystals_dilithium_SOURCE_DIR}")

# Third-party code: optimised, and its warnings are not ours to fix
set(_pqc_flags -O3 -w)
if(PPQ_CPU_NATIVE)
    list(APPEND _pqc_flags -march=native -mtune=native)
endif()

set(_pqc_objects)

function(_ppq_pqc_objects name dir)
    add_library(${name} OBJECT ${ARGN})
    target_include_directories(${name} PRIVATE "${_pqc}/${dir}")
    target_compile_options(${name} PRIVATE ${_pqc_flags})
    set_target_properties(${name} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endfunction()

# Portable reference: one SHAKE build shared by the three modes
_ppq_pqc_objects(ppq_pqc_fips202_ref ref "${_pqc}/ref/fips202.c")
list(APPEND _pqc_objects ppq_pqc_fips202_ref)

foreach(mode 2 3 5)
    set(_sources)
    foreach(f sign.c packing.c polyvec.c poly.c ntt.c reduce.c rounding.c symmetric-shake.c)
        list(APPEND _sources "${_pqc}/ref/${f}")
    endforeach()
    _ppq_pqc_objects(ppq_pqc_ref${mode} ref ${_sources})
    target_compile_definitions(ppq_pqc_ref${mode} PRIVATE DILITHIUM_MODE=${mode})
    list(APPEND _pqc_objects ppq_pqc_ref${mode})

    # Our stage shim over the ref internals: built per mode against the ref headers, but
    # it is ParallelPQ code, so with our warnings rather than -w
    add_library(ppq_pqc_stages${mode} OBJECT src/pqc_stages.c)
    target_include_directories(ppq_pqc_stages${mode} PRIVATE "${_pqc}/ref" src)
    target_compile_definitions(ppq_pqc_stages${mode} PRIVATE DILITHIUM_MODE=${mode})
    target_compile_options(ppq_pqc_stages${mode} PRIVATE -O2)
    target_link_libraries(ppq_pqc_stages${mode} PRIVATE ppq_warnings)
    set_target_properties(ppq_pqc_stages${mode} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    list(APPEND _pqc_objects ppq_pqc_stages${mode})
endforeach()

if(PPQ_CPUREF_AVX2)
    set(_avx2_flags -mavx2 -mpopcnt)

    _ppq_pqc_objects(ppq_pqc_fips202_avx2 avx2
        "${_pqc}/avx2/fips202.c" "${_pqc}/avx2/fips202x4.c" "${_pqc}/avx2/f1600x4.S")
    target_compile_options(ppq_pqc_fips202_avx2 PRIVATE ${_avx2_flags})
    list(APPEND _pqc_objects ppq_pqc_fips202_avx2)

    foreach(mode 2 3 5)
        set(_sources)
        foreach(f sign.c packing.c polyvec.c poly.c ntt.S invntt.S pointwise.S shuffle.S
                  consts.c rejsample.c rounding.c symmetric-shake.c)
            list(APPEND _sources "${_pqc}/avx2/${f}")
        endforeach()
        _ppq_pqc_objects(ppq_pqc_avx2_${mode} avx2 ${_sources})
        target_compile_definitions(ppq_pqc_avx2_${mode} PRIVATE DILITHIUM_MODE=${mode})
        target_compile_options(ppq_pqc_avx2_${mode} PRIVATE ${_avx2_flags})
        list(APPEND _pqc_objects ppq_pqc_avx2_${mode})
    endforeach()
endif()

set(_have_openssl 0)
if(PPQ_WITH_OPENSSL)
    find_package(OpenSSL 3.5 QUIET COMPONENTS Crypto)
    if(OpenSSL_FOUND)
        set(_have_openssl 1)
        message(STATUS "CPU reference: OpenSSL ${OPENSSL_VERSION} (${OPENSSL_CRYPTO_LIBRARY})")
    else()
        message(STATUS "CPU reference: OpenSSL 3.5+ not found, building without it")
    endif()
endif()

# One archive holds the wrapper, randombytes() and every pq-crystals object, so the
# linker resolves the library's randombytes() calls without link-order tricks.
set(_objects)
foreach(lib IN LISTS _pqc_objects)
    list(APPEND _objects $<TARGET_OBJECTS:${lib}>)
endforeach()

add_library(ppq_cpuref STATIC
    src/cpuref.cpp
    src/randombytes.cpp
    src/openssl.cpp
    src/generate.cpp
    src/stages.cpp
    src/cpu_verifier.cpp
    ${_objects})
target_include_directories(ppq_cpuref PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ppq_cpuref PUBLIC ppq_host PRIVATE ppq_warnings)
target_compile_definitions(ppq_cpuref PRIVATE
    PPQ_CPUREF_HAVE_AVX2=$<BOOL:${PPQ_CPUREF_AVX2}>
    PPQ_CPUREF_HAVE_OPENSSL=${_have_openssl})

if(_have_openssl)
    target_link_libraries(ppq_cpuref PRIVATE OpenSSL::Crypto)
endif()
