#include "common_methods.h"

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

__device__ bool bin_search(int goal, int *v, int len)
{
    int l = 0;
    int h = len;
    while (l < h)
    {
        int mid = l + ((h - l) >> 1);
        int v_mid = v[mid];
        if (v_mid < goal)
        {
            l = mid + 1;
        }
        else
        {
            h = mid;
        }
    }
    return (l < len) && (v[l] == goal);
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

chrono_cuda::chrono_cuda(std::string mode)
{
        this->mode = mode;
        cudaEventCreate(&this->start);
        cudaEventCreate(&this->stop);       
}

void chrono_cuda::cc_start()
{
    cudaEventRecord(this->start);
}

void chrono_cuda::cc_stop()
{
    cudaEventRecord(this->stop);
    cudaEventSynchronize(this->stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, this->start, this->stop);
    std::cout << this->mode << " time: " << milliseconds << " ms" << std::endl;
}



chrono_cuda::~chrono_cuda()
{
    cudaEventDestroy(this->start);
    cudaEventDestroy(this->stop);
}

