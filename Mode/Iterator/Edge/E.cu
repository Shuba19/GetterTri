#include "../../../Libs/GraphReader.h"
#include "../../../Libs/ChronoCuda.h"
#include "../../../Libs/CudaUtilities.h"
#include <string>
#include <chrono>

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

__device__ __forceinline__ bool bin_search(const int *csr, int s, int e, int key)
{
    int l = s, r = e - 1;
    while (l <= r)
    {
        int m = (l + r) >> 1;
        int v = csr[m];
        if (v == key)
            return true;
        if (v < key)
            l = m + 1;
        else
            r = m - 1;
    }
    return false;
}

__device__ __forceinline__ int upper_bound(const int *csr, int s, int e, int key)
{
    int l = s, r = e;
    while (l < r)
    {
        int m = (l + r) >> 1;
        if (csr[m] <= key)
            l = m + 1;
        else
            r = m;
    }
    return l;
}
__global__ void edge_search_tri(int num_v, int64_t num_e, const int *__restrict__ ofs, const int *__restrict__ csr, const int *__restrict__ s_edge, int *__restrict__ results);

output_t SearchTriangle_Edge_Iterator(int num_v, int64_t n_edges, std::vector<int> &offsets, std::vector<int> &csr, std::vector<int> &s_edge);

int main(int argc, char *argv[])
{

    if (argv[1] == nullptr)
    {
        std::cerr << "Error: No input file provided. Please provide a graph file as an argument." << std::endl;
        return EXIT_FAILURE;
    }
    chrono_cuda timer("Total Time"), read_timer("Read Time");
    timer.cc_start();
    read_timer.cc_start();
    GraphData graph_data = readGraph(argv[1]);
    read_timer.cc_stop(false);
    output_t output_t = SearchTriangle_Edge_Iterator(graph_data.num_v, graph_data.num_edge, graph_data.offsets, graph_data.csr, graph_data.s_edge);
    timer.cc_stop(false);
    output_t.file = argv[1];
    output_t.num_v = graph_data.num_v;
    output_t.num_e = graph_data.num_edge;
    output_t.total_time = timer.elapsed;
    output_t.read_time = read_timer.elapsed;
    output_t.unit_time = "ms";
    output_t.unit_memory = "bytes";
    output_t.memory_total = 0;
    print_output_as_json(output_t);
}

output_t SearchTriangle_Edge_Iterator(int num_v, int64_t n_edges, std::vector<int> &offsets, std::vector<int> &csr, std::vector<int> &s_edge)
{
    cudaSetDevice(0);
    if (n_edges == 0)
        return {0, 0, 0, 0, 0, 0, 0, 0};

    int *d_csr = nullptr, *d_ofs = nullptr, *d_res = nullptr, *d_s_edge = nullptr;
    unsigned long long *d_sum = nullptr;

    unsigned long long tri_count = 0;
    int tot_spazio = offsets.size() * sizeof(int) + csr.size() * sizeof(int) + s_edge.size() * sizeof(int) + n_edges * sizeof(int) + sizeof(unsigned long long);
    size_t s = n_edges * sizeof(int);
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    CHECK(cudaMallocAsync(&d_ofs, offsets.size() * sizeof(int), stream));
    CHECK(cudaMallocAsync(&d_csr, s, stream));
    CHECK(cudaMallocAsync(&d_s_edge, s, stream));
    CHECK(cudaMallocAsync(&d_res, 1 * sizeof(int), stream));
    CHECK(cudaMallocAsync(&d_sum, sizeof(unsigned long long), stream));

    CHECK(cudaMemcpyAsync(d_ofs, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(d_csr, csr.data(), csr.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(d_s_edge, s_edge.data(), s_edge.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
    int64_t n_blocks = (n_edges + 127) / 128;
    dim3 blockDim(128);
    dim3 gridDim(n_blocks);
    chrono_cuda timer("Edge Iterator", stream);
    timer.cc_start();
    cudaFuncSetCacheConfig(edge_search_tri, cudaFuncCachePreferL1);
    edge_search_tri<<<gridDim, blockDim, 0, stream>>>(num_v, n_edges, d_ofs, d_csr, d_s_edge, d_res);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpyAsync(&tri_count, d_res, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaGetLastError());
    CHECK(cudaStreamSynchronize(stream));
    timer.cc_stop(false);
    CHECK(cudaFreeAsync(d_res, stream));
    CHECK(cudaFreeAsync(d_csr, stream));
    CHECK(cudaFreeAsync(d_ofs, stream));
    CHECK(cudaFreeAsync(d_s_edge, stream));
    CHECK(cudaFreeAsync(d_sum, stream));
    output_t output;
    output.triangles = tri_count / 3;
    output.kernel_time = timer.elapsed;
    output.preprocess_time = 0;

    return output;
}

// Default Kernel for edge iterator
__global__ void edge_search_tri(int num_v, int64_t num_e, const int *__restrict__ ofs, const int *__restrict__ csr, const int *__restrict__ s_edge, int *__restrict__ results)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int n_tri = 0;
    if (id < num_e)
    {
        int s_node = s_edge[id];
        int d_node = csr[id];
        int s1 = ofs[s_node], e1 = ofs[s_node + 1];
        int s2 = ofs[d_node], e2 = ofs[d_node + 1];

        s1 = upper_bound(csr, s1, e1, d_node);
        s2 = upper_bound(csr, s2, e2, d_node);

        int len1 = e1 - s1;
        int len2 = e2 - s2;

        int ss, se, ls, le;
        if (len1 <= len2)
        {
            ss = s1;
            se = e1;
            ls = s2;
            le = e2;
        }
        else
        {
            ss = s2;
            se = e2;
            ls = s1;
            le = e1;
        }

        int short_len = se - ss;
        int long_len = le - ls;
        int i = ss, j = ls;
        while (i < se && j < le)
        {
            int a = csr[i], b = csr[j];
            n_tri += (a == b);
            i += (a <= b);
            j += (a >= b);
        }
    }
    n_tri += __shfl_down_sync(0xffffffff, n_tri, 16);
    n_tri += __shfl_down_sync(0xffffffff, n_tri, 8);
    n_tri += __shfl_down_sync(0xffffffff, n_tri, 4);
    n_tri += __shfl_down_sync(0xffffffff, n_tri, 2);
    n_tri += __shfl_down_sync(0xffffffff, n_tri, 1);
    if (threadIdx.x % 32 == 0)
        atomicAdd(results, n_tri);
}
