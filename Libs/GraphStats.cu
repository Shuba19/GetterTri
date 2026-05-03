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
