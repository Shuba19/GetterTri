#include "../../../Libs/GraphReader.h"
#include "../../../Libs/ChronoCuda.h"
#include "../../../Libs/CudaUtilities.h"
#define TESTING false
#define GRAPH_DEVICE true
#define BLOCK_SIZE 128
#define LONG_THRESHOLD 100

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

output_t SearchTriangle_Edge_Iterator(graph_device graph_data, int threshold, int num_threads);
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
    if (tot_spazio < 1024)
        return {"B", static_cast<float>(tot_spazio)};
    else if (tot_spazio < 1024 * 1024)
        return {"KB", tot_spazio / 1024.0f};
    else if (tot_spazio < 1024 * 1024 * 1024)
        return {"MB", tot_spazio / (1024.0f * 1024.0f)};
    else
        return {"GB", tot_spazio / (1024.0f * 1024.0f * 1024.0f)};
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
    GraphData graph_data = readGraph(argv[1]);
    data_timer.cc_stop(false);
    float preprocess_time = degree_order(graph_data);
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
    output = SearchTriangle_Edge_Iterator(g_graph, threshold, BLOCK_SIZE);
    free_graph(g_graph);
    timer.cc_stop(false);
    std::string name = argv[1];
    output.file = name;
    output.total_time = timer.elapsed;
    output.unit_time = "ms";
    output.preprocess_time = preprocess_time;
    output.read_time = data_timer.elapsed;
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
__global__ void edge_search_tri(int num_v, int64_t num_e, const int *__restrict__ ofs, const int *__restrict__ csr, const int *__restrict__ s_edge, unsigned long long *__restrict__ results, int threshold, bool unbalanced)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int n_tri = 0;
    // thread_block block = cg::this_thread_block();
    // auto warp_coop = cg::tiled_partition(block, 32);
    int n_rip = 0;
    int need_help = 0;
    bool use_merge = false;
    int long_len = 0, short_len = 0;
    int ss, se, ls, le;
    if (id < num_e)
    {

        int s_node = __ldg(&s_edge[id]);
        int d_node = __ldg(&csr[id]);
        if (s_node <= d_node)
        {
            int s1 = ofs[s_node], e1 = ofs[s_node + 1];
            int s2 = ofs[d_node], e2 = ofs[d_node + 1];

            if (s1 < e1 && s2 < e2)
            {
                s1 = upper_bound(csr, s1, e1, d_node);
                s2 = upper_bound(csr, s2, e2, d_node);

                int len1 = e1 - s1;
                int len2 = e2 - s2;

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

                long_len = le - ls;
                short_len = se - ss;
            }
        }
    }
    if (!unbalanced)
        if (short_len > 0 && long_len > 0)
        {
            if (long_len <= short_len * 10)
            {
                int i = ss, j = ls;
                int a = __ldg(&csr[i]);
                int b = __ldg(&csr[j]);

                while (i < se && j < le)
                {
                    int next_i = i + (a <= b);
                    int next_j = j + (a >= b);

                    int next_a = (next_i < se) ? __ldg(&csr[next_i]) : INT_MAX;
                    int next_b = (next_j < le) ? __ldg(&csr[next_j]) : INT_MAX;

                    n_tri += (a == b);
                    i = next_i;
                    j = next_j;
                    a = next_a;
                    b = next_b;
                }
            }
            else
            {
                for (int i = ss; i < se; ++i)
                {
                    n_tri += bin_search(csr, ls, le, csr[i]);
                }
            }
        }
    // Il thread è libero, può comunicare con il warp, per aiutare a contare i triangoli degli altri thread del warp, e alleggerire il lavoro dei thread
    // cerca se esistono thread che chiedono aiuto
    // bisogna capire quali effettivamente abbiano bisogno di aiuto,
    /*
        merge_path ha un carico di base bilanciato
        bin_search conviene se è sbilanciato
        ma se merge_path ha un carico considerevole, allora un thread potrebbe aiutarlo
        bisogna valutare eventuali overhead
    */

    n_tri += __shfl_down_sync(0xffffffff, n_tri, 16);
    n_tri += __shfl_down_sync(0xffffffff, n_tri, 8);
    n_tri += __shfl_down_sync(0xffffffff, n_tri, 4);
    n_tri += __shfl_down_sync(0xffffffff, n_tri, 2);
    n_tri += __shfl_down_sync(0xffffffff, n_tri, 1);
    if ((threadIdx.x & 31) == 0)
        atomicAdd(results, n_tri);
}

output_t SearchTriangle_Edge_Iterator(graph_device graph_data, int threshold, int num_threads)
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
    edge_search_tri<<<gridDim, blockDim, 0, stream>>>(graph_data.num_v, graph_data.num_edge, graph_data.d_ofs, graph_data.d_csr, graph_data.d_s_edge, graph_data.d_sum, threshold, false);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpyAsync(&tri_count, graph_data.d_sum, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaGetLastError());
    timer.cc_stop(false);
    output.triangles = tri_count;
    auto [unit, memory_used] = calculate_space_used({graph_data.num_v, graph_data.num_edge, {}, {}, {}});
    output.memory_total = memory_used;
    output.memory_peak = memory_used;
    output.unit_memory = unit;
    output.time_per_threshold[threshold][128] = timer.elapsed;
    return output;
}

/*


__global__ void edge_search_tri(int num_v, int64_t num_e, const int *__restrict__ ofs, const int *__restrict__ csr, const int *__restrict__ s_edge, unsigned long long *__restrict__ results, int threshold)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int n_tri = 0;
    // thread_block block = cg::this_thread_block();
    // auto warp_coop = cg::tiled_partition(block, 32);
    int n_rip = 0;
    bool use_merge = false;int long_len = 0, short_len = 0;
    if (id < num_e)
    {

        int s_node = __ldg(&s_edge[id]);
        int d_node = __ldg(&csr[id]);
        if (s_node <= d_node)
        {
            int s1 = ofs[s_node], e1 = ofs[s_node + 1];
            int s2 = ofs[d_node], e2 = ofs[d_node + 1];

            if (s1 < e1 && s2 < e2)
            {
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

                 long_len = le - ls;
                 short_len = se - ss;

                if (short_len > 0 && long_len > 0)
                {
                    bool use_merge_path  = long_len <= short_len * 10;
                    int rip = 0;

                    if (long_len <= short_len * 10)
                    {
                        int i = ss, j = ls;
                        int a = __ldg(&csr[i]);
                        int b = __ldg(&csr[j]);

                        while (i < se && j < le)
                        {
                            int next_i = i + (a <= b);
                            int next_j = j + (a >= b);

                            int next_a = (next_i < se) ? __ldg(&csr[next_i]) : INT_MAX;
                            int next_b = (next_j < le) ? __ldg(&csr[next_j]) : INT_MAX;

                            n_tri += (a == b);
                            i = next_i;
                            j = next_j;
                            a = next_a;
                            b = next_b;
                        }
                    }
                    else
                    {
                        for (int i = ss; i < se; ++i)
                        {
                            n_tri += bin_search(csr, ls, le, csr[i]);
                        }
                    }
                }
                // Il thread è libero, può comunicare con il warp, per aiutare a contare i triangoli degli altri thread del warp, e alleggerire il lavoro dei thread

                //cerca se esistono thread attivi

                //eventualeme

            }
        }
        n_tri += __shfl_down_sync(0xffffffff, n_tri, 16);
        n_tri += __shfl_down_sync(0xffffffff, n_tri, 8);
        n_tri += __shfl_down_sync(0xffffffff, n_tri, 4);
        n_tri += __shfl_down_sync(0xffffffff, n_tri, 2);
        n_tri += __shfl_down_sync(0xffffffff, n_tri, 1);
    }
    if ((threadIdx.x & 31) == 0)
        atomicAdd(results, n_tri);
}

*/