#include "GraphStats.h"
#define THRESHOLD_MAX 100
#define THRESHOLD_HELP 5000
#define BLOCK_SIZE 128

__global__ void deg_maj(int num_v, int *off, int *deg_maj)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int d = 0;
    if (id < num_v)
    {
        int offset = off[id + 1] - off[id];
        d = offset > THRESHOLD_MAX ? 1 : 0;
    }
    d += __shfl_down_sync(0xFFFFFFFF, d, 16);
    d += __shfl_down_sync(0xFFFFFFFF, d, 8);
    d += __shfl_down_sync(0xFFFFFFFF, d, 4);
    d += __shfl_down_sync(0xFFFFFFFF, d, 2);
    d += __shfl_down_sync(0xFFFFFFFF, d, 1);
    if ((threadIdx.x & 0x1F) == 0)
        atomicAdd(deg_maj, d);
}

GraphStats calculate_stats_graph(const graph_device &device, const GraphData &graph_data)
{
    GraphStats stats;
    int *d_deg_maj;
    cudaMalloc(&d_deg_maj, sizeof(int));
    cudaMemset(d_deg_maj, 0, sizeof(int));
    int num_blocks = (device.num_v + BLOCK_SIZE - 1) / BLOCK_SIZE;
    deg_maj<<<num_blocks, BLOCK_SIZE>>>(device.num_v, device.d_ofs, d_deg_maj);
    int h_deg_maj;
    cudaMemcpy(&h_deg_maj, d_deg_maj, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_deg_maj);
    stats.deg_maj_threshold = h_deg_maj;
    stats.need_help = h_deg_maj > THRESHOLD_HELP;

    int cur_deg_max = 0;
    for (int i = 0; i < graph_data.num_v; i++)
    {
        int offset = graph_data.offsets[i + 1] - graph_data.offsets[i];
        cur_deg_max = max(cur_deg_max, offset);
    }
    stats.deg_max = cur_deg_max;
    stats.log_maj_deg_max = log2(stats.deg_max);
    return stats;
}

int linear_reg_threshold(const GraphStats &stats)
{
    // implementazione reg lineare stats
    return 0;
}
// reorder graph

bool is_possible_on_gpu(int num_edge, int num_v)
{
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t space_needed_on_device = sizeof(int) * (num_v * 2 + num_edge * 2);
    return space_needed_on_device < free_mem;
}

__global__ void compute_deg(int *off, int num_v, int *deg)
{
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id < num_v)
    {
        deg[id] = off[id + 1] - off[id];
    }
}

__global__ void reassign_id(int num_v, int *off, int *csr, int *new_off, int *new_csr)
{
}


void graph_order_cpu(GraphData &graph_data)
{
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
}

void reorder_graph(GraphData &graph)
{
    cudaSetDevice(0);
    int new_num_e = graph.num_edge / 2;
    int *d_deg, *d_off;
    std::vector<int> ordered_deg(graph.num_v);
    cudaMalloc(&d_off, sizeof(int) * (graph.num_v + 1));
    cudaMalloc(&d_deg, sizeof(int) * graph.num_v);

    cudaMemcpy(d_off, graph.offsets.data(), graph.offsets.size(), cudaMemcpyHostToDevice);
    cudaMemset(d_deg, 0, graph.num_v * sizeof(int));
    dim3 grid((graph.num_v + 255) / 256);
    compute_deg<<<grid, 256>>>(d_off, graph.num_v, d_deg);
    cudaDeviceSynchronize();
    cudaMemcpy(ordered_deg.data(), d_deg, graph.num_v * sizeof(int), cudaMemcpyDeviceToHost);
    std::vector<std::pair<int, int>> degree_vertex(graph.num_v);
    for (int i = 0; i < graph.num_v; ++i)
        degree_vertex[i] = {ordered_deg[i], i};
    std::sort(degree_vertex.begin(), degree_vertex.end());
    std::vector<int> new_id(graph.num_v);
    for (int i = 0; i < graph.num_v; ++i)
        new_id[degree_vertex[i].second] = i;

    // check se la matrice ci sta nella gpu: si -> viene eseguita sulla gpu ; no -> viene eseguita in cpu
    if (is_possible_on_gpu(graph.num_edge, graph.num_v))
    {
        // implementazione gpu
        int *d_csr, *d_new_csr, *d_s_edge, *d_new_s_edge, *d_new_off;
    }
    else
    {
        graph_order_cpu(graph);
    }
    cudaFree(d_off);
    cudaFree(d_deg);
}
