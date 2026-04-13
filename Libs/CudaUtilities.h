#ifndef CUDA_UTILITIES_H
#define CUDA_UTILITIES_H
#include <iostream>
#include <cuda_runtime.h>
#include <vector>



__device__ int searchSourceNode(const int *ofs, int n, int id);
__device__ __forceinline__ bool bin_search(const int *csr, int s, int e, int key);

__global__ void reduce_vector(int64_t num_e, int *d_res, unsigned long long *d_sum);
__global__ void reduce_vector(int64_t num_e, int64_t *d_res, unsigned long long *d_sum);
__global__ void reduce_vector(int64_t num_e, unsigned long long *d_res, unsigned long long *d_sum);
__global__ void reduce_vector(int num_e, int *d_res, unsigned long long *d_sum);
__global__ void prefix_sum(int64_t num_e, int *d_res, unsigned long long *d_sum);

#endif