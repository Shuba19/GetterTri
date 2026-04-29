#include "CudaUtilities.h"
#include "GraphReader.h"

#define CHECK(call)                                                         \
    {                                                                       \
        const cudaError_t error = call;                                     \
        if (error != cudaSuccess)                                           \
        {                                                                   \
            printf("Error %s : %d\n", __FILE__, __LINE__);                  \
            printf("code:%d, reason:%s", error, cudaGetErrorString(error)); \
            exit(1);                                                        \
        }                                                                   \
    }


void load_graph(const GraphData &graph_data, graph_device &g_graph)
{
    int *d_csr = nullptr, *d_ofs = nullptr, *d_s_edge = nullptr;
    int64_t n_edges = graph_data.num_edge;
    unsigned long long *d_sum = nullptr;
    size_t s = n_edges * sizeof(int);
    CHECK(cudaMalloc(&d_ofs, graph_data.offsets.size() * sizeof(int)));
    CHECK(cudaMalloc(&d_csr, s));
    CHECK(cudaMalloc(&d_s_edge, s));
    CHECK(cudaMalloc(&d_sum, sizeof(unsigned long long)));

    CHECK(cudaMemcpy(d_ofs, graph_data.offsets.data(), graph_data.offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_csr, graph_data.csr.data(), graph_data.csr.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_s_edge, graph_data.s_edge.data(), graph_data.s_edge.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_sum, 0, sizeof(unsigned long long)));
    CHECK(cudaDeviceSynchronize());
    g_graph.num_v = graph_data.num_v;
    g_graph.num_edge = graph_data.num_edge;
    g_graph.d_ofs = d_ofs;
    g_graph.d_csr = d_csr;
    g_graph.d_s_edge = d_s_edge;
    g_graph.d_sum = d_sum;
}

void free_graph(graph_device &g_graph)
{
    CHECK(cudaFree(g_graph.d_ofs));
    CHECK(cudaFree(g_graph.d_csr));
    CHECK(cudaFree(g_graph.d_s_edge));
    CHECK(cudaFree(g_graph.d_sum));
}

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