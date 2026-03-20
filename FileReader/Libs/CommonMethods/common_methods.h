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
#include <cuda_fp8.h>

#include <omp.h>
#include <mutex>

struct tiles
{
    double tile[256];
};

struct tiles_b
{
    u_int16_t tile[16];
};
typedef int64_t out_type;

void filter_per_deg(const std::vector<int> &ofs,const std::vector<int> &s_edges,std::vector<int> &thread_level,std::vector<int> &warp_level);

__device__ int searchSourceNode(const int *ofs, int n, int id);
__device__ bool bin_search(int goal, int *v, int len);
__device__ int triangular_col_from_id(int id);

__global__ void reduce_vector(int64_t num_e, int *d_res, unsigned long long *d_sum);
__global__ void reduce_vector(int64_t num_e, int64_t *d_res, unsigned long long *d_sum);
__global__ void reduce_vector(int64_t num_e, unsigned long long *d_res, unsigned long long *d_sum);

out_type SearchTriangle_Edge_Iterator(int num_v,int64_t n_edges, std::vector<int>& offsets, std::vector<int>& csr, std::vector<int>& s_edge);
out_type adaptive_edge_search(int num_v, int64_t n_edges, std::vector<int> &offsets, std::vector<int> &csr, std::vector<int> &s_edge, std::vector<int> &th_level, std::vector<int> &warp_level);

out_type SearchTriangle_Node_Iterator(int num_v,int64_t n_edges, std::vector<int>& csr_size, std::vector<int>& csr, bool undirect);

out_type TTC(int num_v,int64_t n_edges, std::vector<int>offsets, std::vector<int> csr);

out_type TTC_2(int num_v, int64_t n_edges, std::vector<int> offsets, std::vector<int> csr);
out_type TTC_3(int num_v, int64_t n_edges, std::vector<int> offsets, std::vector<int> csr);
out_type triangle_couting_CPU(int num_v,int64_t n_edges,const std::vector<int>& offsets,const std::vector<int>& csr);    

class chrono_cuda
{
    cudaEvent_t start, stop;
    std::string mode;

public:
    chrono_cuda(std::string mode);
    void cc_start();
    void cc_stop();
    ~chrono_cuda();
};

#endif