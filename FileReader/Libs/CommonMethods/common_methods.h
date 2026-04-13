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

#define TILE_SIDE 16
#define TILE_BUILDER_THREADS 128
#define TILE_ROWS_PER_GROUP 16
#define TILE_GROUPS_PER_BLOCK (TILE_BUILDER_THREADS / TILE_ROWS_PER_GROUP)
struct tiles
{
    double tile[256];
};

struct tiles_b
{
    u_int16_t tile[16] = {0};
};
typedef int64_t out_type;

void filter_per_deg(const std::vector<int> &ofs,const std::vector<int> &s_edges,std::vector<int> &thread_level,std::vector<int> &warp_level);

__device__ int searchSourceNode(const int *ofs, int n, int id);
__device__ bool bin_search(int goal, int *v, int len);
__device__ int triangular_col_from_id(int id);

__global__ void tiles_builder(int tpr, int num_v, int total_t, const int *__restrict__ csr, const int *__restrict__ ofs, tiles_b *__restrict__ matrix);


__global__ void reduce_vector(int64_t num_e, int *d_res, unsigned long long *d_sum);
__global__ void reduce_vector(int64_t num_e, int64_t *d_res, unsigned long long *d_sum);
__global__ void reduce_vector(int64_t num_e, unsigned long long *d_res, unsigned long long *d_sum);
__global__ void reduce_vector(int num_e, int *d_res, unsigned long long *d_sum);
__global__ void prefix_sum(int64_t num_e, int *d_res, unsigned long long *d_sum);


out_type SearchTriangle_Edge_Iterator(int num_v,int64_t n_edges, std::vector<int>& offsets, std::vector<int>& csr, std::vector<int>& s_edge);
out_type adaptive_edge_search(int num_v, int64_t n_edges, std::vector<int> &offsets, std::vector<int> &csr, std::vector<int> &s_edge, std::vector<int> &th_level, std::vector<int> &warp_level);

out_type SearchTriangle_Node_Iterator(int num_v,int64_t n_edges, std::vector<int>& csr_size, std::vector<int>& csr, bool undirect);


template <int blockSize> __global__ void reduce6(int *g_in_data, int *g_out_data, unsigned int n);
//TENSOR MODE

__device__ void print_tile(tiles_b tile);
__device__ void print_tile_int(int *tile); 
__device__ void print_tile_uint8(uint8_t *tile);
out_type TTC(int num_v,int64_t n_edges, std::vector<int>offsets, std::vector<int> csr);
out_type TTC_2(int num_v, int64_t n_edges, std::vector<int> offsets, std::vector<int> csr);
out_type TTC_3(int num_v, int64_t n_edges, std::vector<int> offsets, std::vector<int> csr);
out_type TTC_4(int num_v, int64_t n_edges, const std::vector<tiles_b>& tiles, const std::vector<int>& v_tiles);
//TENSOR UTILITIES

__device__ int from_x_y_to_id(int x, int y);

//CPU
out_type triangle_couting_CPU(int num_v,int64_t n_edges,const std::vector<int>& offsets,const std::vector<int>& csr);    

class chrono_cuda
{
    cudaEvent_t start, stop;
    cudaStream_t stream;
    std::string mode;

public:
    chrono_cuda(std::string mode);
    chrono_cuda(std::string mode, cudaStream_t stream);
    void cc_start();
    void cc_stop();
    ~chrono_cuda();
};

#endif