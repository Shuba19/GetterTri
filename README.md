# Counting Triangle Project
This is a project made for counting triangles in a graph using parallellization with CUDA.
In this project, I used different approaches to count triangles in a graph.
To run the code, you need to have a machine equipped with a NVIDIA GPU, and cuda toolkit to compile the code.

## Quick Start Guide

### Prerequisites

- CMake 3.20+
- C++ compiler with C++17 support
- NVIDIA CUDA Toolkit
- NVIDIA GPU (set the correct SM architecture during configure)

### Building

#### 1. Clone the repository

```shell
git clone https://github.com/Shuba19/GetterTri.git
cd GetterTri/
```

#### 2. Create build directory

```shell
mkdir build && cd build
```

#### 3. Configure CMake

```shell
# Example for Ada (sm_89)
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=89 \
      ..
```

Common NVIDIA architecture codes:
- 75 (Turing)
- 80, 86 (Ampere)
- 89 (Ada)
- 90 (Hopper)

#### 4. Build

Build all targets:

```shell
make -j$(nproc)
```

Build one target only:

```shell
make e
# available targets: e, ef, n
```

#### 5. Run

```shell
./bin/e /path/to/graph.txt
```

#### 6. Clean

```shell
make clean
# or full cleanup from repository root:
cd .. && rm -rf build
```



## Example
```shell
  ./bin/ef ../../dataset/road_usa/road_usa.graph
```
# Different Approaches
## Iterators
There are have been developed different versions of iterators, the mainly two are 

## Logic Behind Different Approaches
### Edge Iterator
In this approach, we iterate over each edge in the graph and for each edge (u, v), we find the common neighbors of u and v. The number of common neighbors gives the number of triangles that include the edge (u, v). This approach is efficient for sparse graphs.
### Node Iterator
In this approach, we iterate over each node in the graph and for each node u, we find all pairs of neighbors (v, w) of u. If there is an edge between v and w, then (u, v, w) forms a triangle. This approach is more efficient for dense graphs.
<!-- ### Tensor Core Matrix Multiplication
Since tensor cores are limited to a maximum of 16x16 matrix multiplication, I tried to use tile logic, dividing the CSR matrix into 16x16, eventually padding the matrix to fit into 16x16 tiles.
Then, I multiplied the tiles using tensor cores, and finally summed the resulting matrix's diagonal to get the number of triangles.
![tiles multiplication](https://images.squarespace-cdn.com/content/v1/5a8dbb09bff2006c33266320/1538285346855-38J4GKOCJFYBZMMGB230/gemmtile%281%29.gif?format=1000w) -->

## Performance Comparison
The performance of each approach may vary depending on the characteristics of the input graph (e.g., size, density).
In general, the Tensor Core Matrix Multiplication approach is expected to be the fastest for large and dense graphs, while the Edge Iterator and Node Iterator approaches may perform better for smaller or sparser graphs.
The CPU OpenMP approach provides a good alternative for systems without GPU support, although it may not match the performance of GPU-based methods.



## Authors
- [Shuba19](https://github.com/Shuba19)
