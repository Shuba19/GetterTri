#ifndef COMMON_METHODS_H
#define COMMON_METHODS_H
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <stdio.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

struct tiles
{
    double tile[256];
};

struct tiles_b
{
    u_int16_t tile[16];
};
typedef int64_t out_type;
__device__ int searchSourceNode(const int *ofs, int n, int id);
__device__ bool bin_search(int goal, int *v, int len);
__device__ int triangular_col_from_id(int id);

__global__ void reduce_vector(int64_t num_e, int *d_res, unsigned long long *d_sum);
__global__ void reduce_vector(int64_t num_e, int64_t *d_res, unsigned long long *d_sum);

out_type SearchTriangle_Edge_Iterator(int num_v,int64_t n_edges, std::vector<int>& offsets, std::vector<int>& csr, bool undirect);

out_type SearchTriangle_Node_Iterator(int num_v,int64_t n_edges, std::vector<int>& csr_size, std::vector<int>& csr, bool undirect);

out_type TTC(int num_v,int64_t n_edges, std::vector<int>offsets, std::vector<int> csr);

#endif