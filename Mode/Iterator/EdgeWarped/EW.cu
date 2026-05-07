#include "../../../Libs/GraphReader.h"
#include "../../../Libs/ChronoCuda.h"
#include "../../../Libs/CudaUtilities.h"
#include <cooperative_groups.h>

#define TESTING false
#define GRAPH_DEVICE true
#define BLOCK_SIZE 128
#define COOP_SIZE 32
#define WORK_LOAD_HEAVY 300
#define THRESHOLD_DEGREE 300
#define THRESHOLD_DEGREE_HEAVY 2000
#define LIGHT_MODE true
#define ONLY_BINARY false

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

using namespace cooperative_groups;

namespace cg = cooperative_groups;
__device__ __forceinline__ bool bin_search(const int *__restrict__ csr, int s, int e, int key)
{
    while (s < e)
    {
        int m = s + ((e - s) >> 1);
        int v = __ldg(&csr[m]);
        if (v == key)
            return true;
        s = (v < key) ? m + 1 : s;
        e = (v < key) ? e : m;
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

output_t SearchTriangle_Edge_Iterator(graph_device graph_data, int threshold, int num_threads, bool is_heavy);
float degree_order(GraphData &graph_data);
int get_device_memory()
{
    size_t free_mem, total_mem;
    CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    return static_cast<int>(total_mem / (1024 * 1024)); // in MB
}

std::pair<std::string, float> calculate_space_used(const GraphData &graph_data)
{
    int64_t tot_spazio = graph_data.num_v * sizeof(int) + graph_data.num_edge * sizeof(int) + graph_data.num_edge * sizeof(int) + graph_data.num_edge * sizeof(int) + sizeof(int64_t);
    return {"MB", static_cast<float>(tot_spazio) / (1024 * 1024)};
}

bool can_run_on_gpu(const GraphData &g)
{
    auto [unit, mem_used] = calculate_space_used(g);
    if (unit == "GB")
        mem_used *= 1024;
    if (unit == "KB")
        mem_used /= 1024;
    if (unit == "B")
        mem_used /= (1024 * 1024);
    int device_mem = get_device_memory();
    return mem_used < device_mem * 0.8;
}

int main(int argc, char *argv[])
{
    chrono_cuda timer("Total Execution"), data_timer("File Reading");
    timer.cc_start();

    if (argv[1] == nullptr)
    {
        std::cerr << "Error: No input file provided. Please provide a graph file as an argument." << std::endl;
        return EXIT_FAILURE;
    }
    data_timer.cc_start();
    GraphData graph_data;
    if (0)
        graph_data = readGraph(argv[1]);
    else
        graph_data = readGraph_Forward(argv[1]);
    data_timer.cc_stop(false);
    float preprocess_time = 0; //degree_order(graph_data);
    int counter = 0;
    for (int i = 0; i < graph_data.num_v; ++i)
    {
        int deg = graph_data.offsets[i + 1] - graph_data.offsets[i];
        if (deg > THRESHOLD_DEGREE)
            counter++;
    }
    bool is_heavy = counter > THRESHOLD_DEGREE_HEAVY;
    output_t output;

    if (!can_run_on_gpu(graph_data))
    {
        output.file = argv[1];
        output.total_time = preprocess_time;
        output.unit_time = "ms";
        output.triangles = -1;
        print_output_as_json(output);
        return EXIT_FAILURE;
    }

    int threshold = 10;
    graph_device g_graph;
    load_graph(graph_data, g_graph);
    output = SearchTriangle_Edge_Iterator(g_graph, threshold, BLOCK_SIZE, is_heavy);
    free_graph(g_graph);
    timer.cc_stop(false);
    std::string name = argv[1];
    output.file = name;
    output.total_time = timer.elapsed;
    output.unit_time = "ms";
    output.preprocess_time = preprocess_time;
    output.read_time = data_timer.elapsed;
    output.num_v = graph_data.num_v;
    output.num_e = graph_data.num_edge;
    print_output_as_json(output);
    return 0;
}

float degree_order(GraphData &graph_data)
{
    chrono_cuda timer("Degree Ordering");
    timer.cc_start();
    int n = graph_data.num_v;

    std::vector<std::pair<int, int>> degree_vertex(n);
    for (int i = 0; i < n; ++i)
        degree_vertex[i] = {graph_data.offsets[i + 1] - graph_data.offsets[i], i};
    std::sort(degree_vertex.begin(), degree_vertex.end());

    std::vector<int> new_id(n);
    for (int i = 0; i < n; ++i)
        new_id[degree_vertex[i].second] = i;

    std::vector<int> new_csr;
    std::vector<int> new_s_edge;
    std::vector<int> new_offsets(n + 1, 0);

    new_csr.reserve(graph_data.num_edge / 2);
    new_s_edge.reserve(graph_data.num_edge / 2);

    for (int new_src = 0; new_src < n; ++new_src)
    {
        int old_src = degree_vertex[new_src].second;
        int old_start = graph_data.offsets[old_src];
        int old_end = graph_data.offsets[old_src + 1];
        int start_idx = static_cast<int>(new_csr.size());

        for (int j = old_start; j < old_end; ++j)
        {
            int new_neighbor = new_id[graph_data.csr[j]];
            if (new_neighbor > new_src)
            {
                new_csr.push_back(new_neighbor);
                new_s_edge.push_back(new_src);
            }
        }
        std::sort(new_csr.begin() + start_idx, new_csr.end());
        new_offsets[new_src + 1] = static_cast<int>(new_csr.size());
    }
    graph_data.offsets = std::move(new_offsets);
    graph_data.csr = std::move(new_csr);
    graph_data.s_edge = std::move(new_s_edge);
    graph_data.num_edge = graph_data.csr.size();
    timer.cc_stop(false);
    return timer.elapsed;
}

// Default Kernel for edge iterator
__global__ void edge_search_tri(int num_v, int64_t num_e, const int *__restrict__ ofs, const int *__restrict__ csr, const int *__restrict__ s_edge,
                                int *__restrict__ results, bool is_heavy_graph)
{
    cg::thread_block_tile<COOP_SIZE> warp = cg::tiled_partition<COOP_SIZE>(cg::this_thread_block());
    int lane = warp.thread_rank();
    __shared__ int search_node[4][128];
    int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int n_tri = 0;
    int tot_operations = 0;
    int ss = 0, se = 0, ls = 0, le = 0;
    bool has_work = false;
    bool is_heavy = false;
    bool merge_path = false;
    if (id < num_e)
    {
        int u = __ldg(&s_edge[id]);
        int v = __ldg(&csr[id]);

        if (u > v)
        {
            int u_s = ofs[u], u_e = ofs[u + 1];
            int v_s = ofs[v], v_e = ofs[v + 1];

            if (u_e > u_s && v_e > v_s)
            {
                if ((u_e - u_s) <= (v_e - v_s))
                {
                    ss = u_s;
                    se = u_e;
                    ls = v_s;
                    le = v_e;
                }
                else
                {
                    ss = v_s;
                    se = v_e;
                    ls = u_s;
                    le = u_e;
                }
                float O_bin_search, O_merge_path;
                has_work = true;
                int long_len = le - ls;
                int short_len = se - ss;
                int N = long_len + short_len;
                O_bin_search = (short_len)*__log2f(long_len);
                O_merge_path = __log10f(N) + (N);
                merge_path = O_merge_path < O_bin_search;
                int num_rep = merge_path ? O_bin_search : O_merge_path;
                float warp_rep = num_rep / COOP_SIZE;
                tot_operations = num_rep;
            }
        }
    }
    tot_operations += __shfl_down_sync(0xffffffff, tot_operations, 16);
    tot_operations += __shfl_down_sync(0xffffffff, tot_operations, 8);
    tot_operations += __shfl_down_sync(0xffffffff, tot_operations, 4);
    tot_operations += __shfl_down_sync(0xffffffff, tot_operations, 2);
    tot_operations += __shfl_down_sync(0xffffffff, tot_operations, 1);
    is_heavy = (tot_operations > WORK_LOAD_HEAVY);
#if LIGHT_MODE
    // se light, svolge lavoro
    if (has_work && !is_heavy)
    {
        if (merge_path)
        {
            int i = ss, j = ls;
            int a = __ldg(&csr[i]);
            int b = __ldg(&csr[j]);
            while (i < se && j < le)
            {
                int next_i = i + (a <= b);
                int next_j = j + (a >= b);
                n_tri += (a == b);
                i = next_i;
                j = next_j;
                a = (next_i < se) ? __ldg(&csr[next_i]) : INT_MAX;
                b = (next_j < le) ? __ldg(&csr[next_j]) : INT_MAX;
            }
        }
        else
        {
            for (int i = ss; i < se; ++i)
                n_tri += bin_search(csr, ls, le, __ldg(&csr[i]));
        }
        has_work = false;
    }
#endif
    // coop
    uint32_t heavy_mask = warp.ballot(has_work && is_heavy);

    while (heavy_mask > 0)
    {
        int leader = __ffs(heavy_mask) - 1;

        int shared_ss = warp.shfl(ss, leader);
        int shared_se = warp.shfl(se, leader);
        int shared_ls = warp.shfl(ls, leader);
        int shared_le = warp.shfl(le, leader);
        bool shared_merge_path = warp.shfl(merge_path, leader);

        int short_len = shared_se - shared_ss;
#if !ONLY_BINARY
        if (shared_merge_path)
        {
            int short_len = shared_se - shared_ss;
            int chunk = (short_len + COOP_SIZE - 1) / COOP_SIZE;
            int i = shared_ss + lane * chunk;
            int i_end = min(i + chunk, shared_se);

            if (i < shared_se)
            {
                int key = __ldg(&csr[i]);
                int lo = shared_ls, hi = shared_le;
                while (lo < hi)
                {
                    int mid = lo + ((hi - lo) >> 1);
                    if (__ldg(&csr[mid]) < key)
                        lo = mid + 1;
                    else
                        hi = mid;
                }
                int j = lo;

                int a = key;
                int b = (j < shared_le) ? __ldg(&csr[j]) : INT_MAX;

                while (i < i_end & j < shared_le)
                {
                    bool adv_i = (a <= b);
                    bool adv_j = (a >= b);
                    n_tri += (a == b);
                    i += adv_i;
                    j += adv_j;
                    a = adv_i & (i < i_end) ? __ldg(&csr[i]) : a;
                    b = adv_j & (j < shared_le) ? __ldg(&csr[j]) : b;
                }
            }
        }
        else
            for (int i = shared_ss + lane; i < shared_se; i += COOP_SIZE)
                n_tri += bin_search(csr, shared_ls, shared_le, __ldg(&csr[i]));

#else

#pragma unroll
        for (int i = shared_ss + lane; i < shared_se; i += COOP_SIZE)
            n_tri += bin_search(csr, shared_ls, shared_le, __ldg(&csr[i]));
#endif
        heavy_mask &= ~(1u << leader);

        if (lane == leader)
            has_work = false;
    }

    n_tri += __shfl_down_sync(0xffffffff, n_tri, 16);
    n_tri += __shfl_down_sync(0xffffffff, n_tri, 8);
    n_tri += __shfl_down_sync(0xffffffff, n_tri, 4);
    n_tri += __shfl_down_sync(0xffffffff, n_tri, 2);
    n_tri += __shfl_down_sync(0xffffffff, n_tri, 1);

    if ((threadIdx.x & 31) == 0)
        atomicAdd(results, n_tri);
}

output_t SearchTriangle_Edge_Iterator(graph_device graph_data, int threshold, int num_threads, bool is_heavy)
{
    cudaSetDevice(0);
    if (graph_data.num_edge == 0)
        return output_t();
    output_t output;
    int64_t tri_count = 0;
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    int64_t n_blocks = (graph_data.num_edge + num_threads - 1) / num_threads;
    dim3 blockDim(num_threads);
    dim3 gridDim(n_blocks);
    chrono_cuda timer("Edge Iterator", stream);
    timer.cc_start();
    cudaFuncSetCacheConfig(edge_search_tri, cudaFuncCachePreferL1);
    edge_search_tri<<<gridDim, blockDim, 0, stream>>>(graph_data.num_v, graph_data.num_edge, graph_data.d_ofs, graph_data.d_csr, graph_data.d_s_edge, graph_data.d_sum, false);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpyAsync(&tri_count, graph_data.d_sum, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaGetLastError());
    timer.cc_stop(false);
    output.triangles = tri_count;
    auto [unit, memory_used] = calculate_space_used({graph_data.num_v, graph_data.num_edge, {}, {}, {}});
    output.memory_total = memory_used;
    output.memory_peak = memory_used;
    output.unit_memory = unit;
    output.kernel_time = timer.elapsed;
    return output;
}