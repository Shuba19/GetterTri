# Counting Triangle Project
This is a project made for counting triangles in a graph using parallellization with CUDA.
In this project, I used different approaches to count triangles in a graph.
To run the code, you need to have a machine equipped with a NVIDIA GPU, and cuda toolkit to compile the code.

## Structure of the Folder
The folder is structured as follows:
```text
.
├── .gitignore
├── LICENSE
├── Main.cu
├── Makefile
├── README.md
└── FileReader
    ├── command_args.cpp
    ├── command_args.h
    ├── FileReader.cu
    ├── FileReader.h
    └── Libs
        ├── CommonMethods
        │   ├── common_methods.cu
        │   └── common_methods.h
        ├── CUDA_Tri_Edge_Iterator
        │   └── EdgeIterator.cu
        ├── CUDA_Tri_Node_Iterator
        │   └── NodeIterator.cu
        ├── CUDA_Tri_Tensor_Multi
        │   └── TensorCalculation.cu
        └── OpenMPTriCalc
            └── TriangleCountingCPU.cpp
```
## Compilation
To copy the code to your local machine, use the following command:
```bash
git clone https://github.com/Shuba19/Cuda-Triangle-Counting.git
cd Cuda-Triangle-Counting
```
To compile the code run the ``` make ``` command in the terminal
## Run
To run the code, use the following command in the terminal:
```bash
./out  <arguments>
```
### Arguments
- -i : Input file path (in a METIS style format, look at the  [manual](https://sites.cc.gatech.edu/dimacs10/data/manual.ps) for further information)
- -d : Specify if the graph is directed, if there is no such flag, the graph is considered undirected
- -h : To display help message
- -help : To display help message
- -b : To run the code in benchmarking mode
- -v : Verbose mode
- -mode : To specify which approach to use for counting triangles. Possible values are:
    - 0: Edge Iterator
    - 1: Node Iterator
    - 2: Tensor Core Matrix Multiplication
    - 3: CPU OpenMP
### Example
An example of running the code is as follows:
```bash
./out -i input.graph -v -mode 1
```
In this way, the code will take the input graph from the file "input.graph", count the triangles using the Node Iterator approach, and display the time taken to count the triangles.

  
## Logic Behind Different Approaches
### Edge Iterator
In this approach, we iterate over each edge in the graph and for each edge (u, v), we find the common neighbors of u and v. The number of common neighbors gives the number of triangles that include the edge (u, v). This approach is efficient for sparse graphs.
### Node Iterator
In this approach, we iterate over each node in the graph and for each node u, we find all pairs of neighbors (v, w) of u. If there is an edge between v and w, then (u, v, w) forms a triangle. This approach is more efficient for dense graphs.
### Tensor Core Matrix Multiplication
Since tensor cores are limited to a maximum of 16x16 matrix multiplication, I tried to use tile logic, dividing the CSR matrix into 16x16, eventually padding the matrix to fit into 16x16 tiles.
Then, I multiplied the tiles using tensor cores, and finally summed the resulting matrix's diagonal to get the number of triangles.
![tiles multiplication](https://images.squarespace-cdn.com/content/v1/5a8dbb09bff2006c33266320/1538285346855-38J4GKOCJFYBZMMGB230/gemmtile%281%29.gif?format=1000w)
### CPU OpenMP
In this approach, I used OpenMP to parallelize the Node Iterator approach on the CPU.
This approach is useful for machines without a compatible NVIDIA GPU.

## Performance Comparison
The performance of each approach may vary depending on the characteristics of the input graph (e.g., size, density).
In general, the Tensor Core Matrix Multiplication approach is expected to be the fastest for large and dense graphs, while the Edge Iterator and Node Iterator approaches may perform better for smaller or sparser graphs.
The CPU OpenMP approach provides a good alternative for systems without GPU support, although it may not match the performance of GPU-based methods.



## Authors
- [Shuba19](https://github.com/Shuba19)