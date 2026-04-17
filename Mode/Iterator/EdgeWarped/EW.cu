#include "../../../Libs/GraphReader.h"
#include "../../../Libs/ChronoCuda.h"
#include "../../../Libs/CudaUtilities.h"

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

struct output_t
{
    std::string file;
    int triangles, memory_total_mb, memory_peak_mb;
    float total_time, kernel_time;
    std::string unit_time;
};

void print_output_as_json(const output_t &output)
{
    std::cout << "{\n";
    std::cout << "  \"file\": \"" << output.file << "\",\n";
    std::cout << "  \"triangles\": " << output.triangles << ",\n";
    std::cout << "  \"total_time\": " << output.total_time << ",\n";
    std::cout << "  \"kernel_time\": " << output.kernel_time << ",\n";
    std::cout << "  \"memory_total_mb\": " << output.memory_total_mb << ",\n";
    std::cout << "  \"memory_peak_mb\": " << output.memory_peak_mb << ",\n";
    std::cout << "  \"unit_time\": \"" << output.unit_time << "\"\n";
    std::cout << "}\n";
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
__global__ void edge_search_tri(int num_v, int64_t num_e,const int * __restrict__ ofs,const int * __restrict__ csr,const int * __restrict__ s_edge,int * __restrict__ results);

output_t SearchTriangle_Edge_Iterator(int num_v,int64_t n_edges, std::vector<int>& offsets, std::vector<int>& csr, std::vector<int>& s_edge);

int main(int argc, char *argv[])
{
    chrono_cuda timer("Total Execution"), data_timer("File Reading");
    timer.cc_start();

    if(argv[1] == nullptr)
    {
        std::cerr << "Error: No input file provided. Please provide a graph file as an argument." << std::endl;
        return EXIT_FAILURE;
    }
    data_timer.cc_start();
    GraphData graph_data = readGraph_Forward(argv[1]);
    data_timer.cc_stop(false);
    output_t output = SearchTriangle_Edge_Iterator(graph_data.num_v, graph_data.num_edge, graph_data.offsets, graph_data.csr, graph_data.s_edge);
    timer.cc_stop(false);
    output.file = argv[1];
    output.total_time = timer.elapsed;
    output.unit_time = "ms";
    print_output_as_json(output);
    return 0;
}


output_t SearchTriangle_Edge_Iterator(int num_v, int64_t n_edges, std::vector<int> &offsets, std::vector<int> &csr, std::vector<int> &s_edge)
{
    cudaSetDevice(0);
    if (n_edges == 0)
        return output_t();

    output_t output;
    int *d_csr = nullptr, *d_ofs = nullptr, *d_res = nullptr, *d_s_edge = nullptr;
    int *d_sum = nullptr;

    int tri_count = 0;
    int tot_spazio = offsets.size() * sizeof(int) + csr.size() * sizeof(int) + s_edge.size() * sizeof(int) + n_edges * sizeof(int) + sizeof(int);
    float tot_spazio_mb = tot_spazio / (1024.0f * 1024.0f);
    output.memory_total_mb = tot_spazio_mb;
    output.memory_peak_mb = tot_spazio_mb;
    size_t s = n_edges * sizeof(int);
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    CHECK(cudaMallocAsync(&d_ofs, offsets.size() * sizeof(int), stream));
    CHECK(cudaMallocAsync(&d_csr, s, stream));
    CHECK(cudaMallocAsync(&d_s_edge, s, stream));
    CHECK(cudaMallocAsync(&d_res, s, stream));
    CHECK(cudaMallocAsync(&d_sum, sizeof(int), stream));

    CHECK(cudaMemcpyAsync(d_ofs, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(d_csr, csr.data(), csr.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(d_s_edge, s_edge.data(), s_edge.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemsetAsync(d_sum, 0, sizeof(int), stream));
    std::cout << "EDGE COUNT: " << n_edges << std::endl;
    std::cout << "CSR SIZE: " << csr.size() << std::endl;
    int64_t n_blocks = (n_edges + 127) / 128;
    dim3 blockDim(128);
    dim3 gridDim(n_blocks);
    chrono_cuda timer("Edge Iterator", stream);
    timer.cc_start();
    cudaFuncSetCacheConfig(edge_search_tri, cudaFuncCachePreferL1);
    edge_search_tri<<<gridDim, blockDim, 0, stream>>>(num_v, n_edges, d_ofs, d_csr, d_s_edge, d_res);
    CHECK(cudaGetLastError());
    chrono_cuda timer_reduce("Reduction", stream);
    timer_reduce.cc_start();
    reduce_vector<<<gridDim, blockDim, 0, stream>>>(n_edges, d_res, d_sum);
    timer_reduce.cc_stop(false);
    CHECK(cudaMemcpyAsync(&tri_count, d_sum, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaGetLastError());
    timer.cc_stop(false);
    CHECK(cudaFree(d_res));
    CHECK(cudaFree(d_csr));
    CHECK(cudaFree(d_ofs));
    CHECK(cudaFree(d_s_edge));
    CHECK(cudaFree(d_sum));
    output.triangles = tri_count;
    output.kernel_time = timer.elapsed;
    return output;
}


// Default Kernel for edge iterator
__global__ void edge_search_tri(int num_v, int64_t num_e, const int *__restrict__ ofs, const int *__restrict__ csr, const int *__restrict__ s_edge, int *__restrict__ results)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_e)
        return;

    int s_node = s_edge[id];
    int d_node = csr[id];
    if (d_node <= s_node)
    {
        results[id] = 0;
        return;
    }

    int s1 = ofs[s_node], e1 = ofs[s_node + 1];
    int s2 = ofs[d_node], e2 = ofs[d_node + 1];

    s1 = upper_bound(csr, s1, e1, d_node);
    s2 = upper_bound(csr, s2, e2, d_node);

    int len1 = e1 - s1;
    int len2 = e2 - s2;

    int n_tri = 0;

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
    if (long_len <= short_len * 16)
    {
        int i = ss, j = ls;
        while (i < se && j < le)
        {
            int a = csr[i], b = csr[j];
            n_tri += (a == b);
            i += (a <= b);
            j += (a >= b);
        }
    }
    else
    {
        for (int i = ss; i < se; ++i)
        {
            n_tri += bin_search(csr, ls, le, csr[i]);
        }
    }
    
    results[id] = n_tri;
}
