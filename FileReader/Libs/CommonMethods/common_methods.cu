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


chrono_cuda::chrono_cuda(std::string mode)
{
        this->mode = mode;
        cudaEventCreate(&this->start);
        cudaEventCreate(&this->stop);  
        this->stream = 0;     
}

chrono_cuda::chrono_cuda(std::string mode, cudaStream_t stream)
{
        this->mode = mode;
        cudaEventCreate(&this->start);
        cudaEventCreate(&this->stop);  
        this->stream = stream;     
}

void chrono_cuda::cc_start()
{
    cudaEventRecord(this->start, this->stream);
}

void chrono_cuda::cc_stop()
{
    //sync con lo stream
    cudaEventRecord(this->stop, this->stream);
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



template <unsigned int blockSize>__device__ void warpReduce(volatile int* sdata, unsigned int tid){
    if(blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if(blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if(blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if(blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if(blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if(blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

// REDUCTION 6 – Multiple Adds / Threads
template <int blockSize> __global__ void reduce6(int *g_in_data, int *g_out_data, unsigned int n){
    extern __shared__ int sdata[];  // stored in the shared memory

    // Each thread loading one element from global onto shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockDim.x * 2 * gridDim.x;
    sdata[tid] = 0;

    while(i < n) { 
      sdata[tid] += g_in_data[i] + g_in_data[i + blockSize]; 
      i += gridSize; 
    }
    __syncthreads();

    // Perform reductions in steps, reducing thread synchronization
    if (blockSize >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
    }

    if (tid < 32) warpReduce<blockSize>(sdata, tid);

    if (tid == 0){
        g_out_data[blockIdx.x] = sdata[0];
    }
}

template __global__ void reduce6<128>(int *g_in_data, int *g_out_data, unsigned int n);