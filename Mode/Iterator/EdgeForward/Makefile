CMAKE ?= cmake
TARGET := ef

PROJECT_ROOT := $(abspath $(CURDIR)/../../..)
BUILD_DIR ?= $(PROJECT_ROOT)/build

# Override from command line, e.g. make SM=86
SM ?= 89

.PHONY: all run clean

all:
	$(CMAKE) -S "$(PROJECT_ROOT)" -B "$(BUILD_DIR)" \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_CUDA_ARCHITECTURES=$(SM)
	$(CMAKE) --build "$(BUILD_DIR)" --target $(TARGET) -j

# Usage: make run GRAPH=path/to/graph.txt
run: all
	@if [ -z "$(GRAPH)" ]; then \
		echo "Usage: make run GRAPH=path/to/graph.txt"; \
		exit 1; \
	fi
	"$(BUILD_DIR)/bin/$(TARGET)" $(GRAPH)

clean:
	rm -rf "$(BUILD_DIR)"
