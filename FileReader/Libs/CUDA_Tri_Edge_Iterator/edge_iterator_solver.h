#ifndef edge_iterator_solver
#define edge_iterator_solver
#include "../CommonMethods/common_methods.h"
#define WARP_SIZE 32
#define THRESHOLD 1024
__device__ __forceinline__ bool bin_search(const int *csr, int s, int e, int key);

__device__ __forceinline__ int upper_bound(const int *csr, int s, int e, int key);



__global__ void edge_search_tri(int num_v, int64_t num_e,const int * __restrict__ ofs,const int * __restrict__ csr,const int * __restrict__ s_edge,int * __restrict__ results);


__global__ void dynamic_search_tri(int num_v, int64_t num_e,const int * __restrict__ ofs,const int * __restrict__ csr,const int * __restrict__ s_edge,int * __restrict__ results);


__global__ void edge_warp_search_tri(int num_v, int64_t num_e,const int * __restrict__ ofs,const int * __restrict__ csr,const int * __restrict__ s_edge,int * __restrict__ results, const int *warp_level);


__global__ void edge_thread_search_tri(int num_v, int64_t num_e, const int *__restrict__ ofs, const int *__restrict__ csr, const int *__restrict__ s_edge, int *__restrict__ results, const int *thread_level);

__global__ void help_search_tri(int num_v, int64_t num_e,const int * __restrict__ ofs,const int * __restrict__ csr,const int * __restrict__ s_edge,int * __restrict__ results);

#endif