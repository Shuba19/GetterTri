NVCC := nvcc

TARGET := ttc_bhso
SRC := TTC_BHSO.cu \
	../../../Libs/GraphReader.cu \
	../../../Libs/ChronoCuda.cu \
	../../../Libs/CudaUtilities.cu \
	../../../Libs/TensorUtilities.cu

BUILD_DIR := build

# Override from command line, e.g. make SM=86
SM ?= 89

NVCCFLAGS := -O3 -std=c++17 -lineinfo -rdc=true -Xcompiler -Wall -gencode arch=compute_$(SM),code=sm_$(SM)

.PHONY: all run clean

all: $(BUILD_DIR)/$(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/$(TARGET): $(SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

# Usage: make run GRAPH=path/to/graph.txt
run: all
	@if [ -z "$(GRAPH)" ]; then \
		echo "Usage: make run GRAPH=path/to/graph.txt"; \
		exit 1; \
	fi
	./$(BUILD_DIR)/$(TARGET) $(GRAPH)

clean:
	rm -rf $(BUILD_DIR)
