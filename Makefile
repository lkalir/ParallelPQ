detect_platform = $(shell \
    if command -v nvcc >/dev/null 2>&1; then echo nvidia; \
    elif command -v hipconfig >/dev/null 2>&1; then echo amd; fi)

GPU_PLATFORM ?= $(if $(HIP_PLATFORM),$(HIP_PLATFORM),$(detect_platform))
override GPU_PLATFORM := $(patsubst nvcc,nvidia,$(GPU_PLATFORM))
GPU_ARCH       ?= native
BUILD_TYPE     ?= Release
BUILD_DIR      ?= build
HOST_BUILD_DIR ?= build-host
CMAKE          ?= cmake
JOBS           ?= $(shell nproc 2>/dev/null || echo 4)

CACHE    := $(BUILD_DIR)/CMakeCache.txt
EXE      := ppq.exe
RUN_ARGS ?= info

.DEFAULT_GOAL := all
.PHONY: all build configure probe run test test-host clean distclean

all: $(EXE)

.PHONY: $(EXE)
$(EXE): build
	@cp -f "$(BUILD_DIR)/bin/$(EXE)" "$@"

build: configure
	@$(CMAKE) --build "$(BUILD_DIR)" --target ppq --parallel $(JOBS)

probe: configure
	@$(CMAKE) --build "$(BUILD_DIR)" --target probe --parallel $(JOBS)
	@cp -f "$(BUILD_DIR)/bin/probe.exe" probe.exe

configure:
	@command -v $(CMAKE) >/dev/null 2>&1 || { echo "error: cmake not found on PATH" >&2; exit 1; }
	@case "$(GPU_PLATFORM)" in \
	    nvidia) command -v nvcc >/dev/null 2>&1 || { \
	        echo "error: GPU_PLATFORM=nvidia but nvcc is not on PATH" >&2; exit 1; } ;; \
	    amd) ;; \
	    *) echo "error: set HIP_PLATFORM or GPU_PLATFORM to nvidia or amd (got '$(GPU_PLATFORM)')" >&2; \
	       exit 1 ;; \
	esac
	@if [ -f "$(CACHE)" ] && ! grep -qx 'PPQ_GPU_PLATFORM:STRING=$(GPU_PLATFORM)' "$(CACHE)"; then \
	    echo "$(BUILD_DIR)/ was configured for another GPU platform; removing it"; \
	    rm -rf "$(BUILD_DIR)"; \
	fi
	@$(CMAKE) -S . -B "$(BUILD_DIR)" \
	    -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
	    -DPPQ_GPU_PLATFORM=$(GPU_PLATFORM) \
	    -DPPQ_GPU_ARCH="$(GPU_ARCH)"

run: $(EXE)
	./$(EXE) $(RUN_ARGS)

test: configure
	@$(CMAKE) --build "$(BUILD_DIR)" --parallel $(JOBS)
	@cd "$(BUILD_DIR)" && ctest --output-on-failure

test-host:
	@$(CMAKE) -S . -B "$(HOST_BUILD_DIR)" -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DPPQ_GPU_ENABLED=OFF
	@$(CMAKE) --build "$(HOST_BUILD_DIR)" --parallel $(JOBS)
	@cd "$(HOST_BUILD_DIR)" && ctest --output-on-failure

clean:
	@if [ -d "$(BUILD_DIR)" ]; then \
	    $(CMAKE) --build "$(BUILD_DIR)" --target clean 2>/dev/null || true; \
	fi
	@rm -f $(EXE) probe.exe

distclean:
	@rm -rf "$(BUILD_DIR)" "$(HOST_BUILD_DIR)" $(EXE) probe.exe
	@echo "removed $(BUILD_DIR)/, $(HOST_BUILD_DIR)/, $(EXE) and probe.exe"
