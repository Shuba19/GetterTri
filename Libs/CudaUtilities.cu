#include "CudaUtilities.h"

__device__ int searchSourceNode(const int *ofs, int n, int id)
{
    int low = 0, high = n;
    while (low < high)
    {
        int mid = (low + high) >> 1;
        if (ofs[mid] <= id)
            low = mid + 1;
        else
            high = mid;
    }
    return low;
}

__global__ void reduce_vector(int64_t num_e, int *d_res, unsigned long long *d_sum)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    extern __shared__ unsigned long long s_data[];
    if (id < num_e)
        s_data[tid] = (unsigned long long)d_res[id];
    else
        s_data[tid] = 0;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            s_data[tid] += s_data[tid + stride];
        __syncthreads();
    }
    if (tid == 0)
        atomicAdd(d_sum, s_data[0]);
}



__global__ void reduce_vector(int  num_e, int *d_res, unsigned long long *d_sum)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    extern __shared__ unsigned long long s_data[];
    if (id < num_e)
        s_data[tid] = (unsigned long long)d_res[id];
    else
        s_data[tid] = 0;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            s_data[tid] += s_data[tid + stride];
        __syncthreads();
    }
    if (tid == 0)
        atomicAdd(d_sum, s_data[0]);
}
__global__ void reduce_vector(int  num_e, int *d_res, int *d_sum)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    extern __shared__ unsigned long long s_data[];
    if (id < num_e)
        s_data[tid] = (unsigned long long)d_res[id];
    else
        s_data[tid] = 0;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            s_data[tid] += s_data[tid + stride];
        __syncthreads();
    }
    if (tid == 0)
        atomicAdd(d_sum, s_data[0]);
}


__global__ void reduce_vector(int64_t num_e, int64_t *d_res, unsigned long long *d_sum)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    extern __shared__ unsigned long long s_data[];
    if (id < num_e)
        s_data[tid] = (unsigned long long)d_res[id];
    else
        s_data[tid] = 0;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            s_data[tid] += s_data[tid + stride];
        __syncthreads();
    }
    if (tid == 0)
        atomicAdd(d_sum, s_data[0]);
}

__global__ void reduce_vector(int64_t num_e, unsigned long long *d_res, unsigned long long *d_sum)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    extern __shared__ unsigned long long s_data[];
    if (id < num_e)
        s_data[tid] = d_res[id];
    else
        s_data[tid] = 0;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            s_data[tid] += s_data[tid + stride];
        __syncthreads();
    }
    if (tid == 0)
        atomicAdd(d_sum, s_data[0]);
}

__global__ void prefix_sum(int64_t num_e, int *d_res, unsigned long long *d_sum)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    extern __shared__ unsigned long long s_data[];
    if (id < num_e)
        s_data[tid] = (unsigned long long)d_res[id];
    else
        s_data[tid] = 0;
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride <<= 1)
    {
        unsigned long long val = 0;
        if (tid >= stride)
            val = s_data[tid - stride];
        __syncthreads();
        s_data[tid] += val;
        __syncthreads();
    }
    if (id < num_e)
        d_res[id] = (int)s_data[tid];
}