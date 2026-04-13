CXX := nvcc -O2
ARCH := -arch=sm_89
CXXFLAGS := -std=c++17 -O2

TARGET := sparse_tile_converter
SRC := SparseTileConverter.cu
.PHONY: all build run clean help

all: build

build: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(ARCH) $< -o $@

clean:
	rm -f $(TARGET)
