#include "edge_iterator_solver.h"
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

out_type SearchTriangle_Edge_Iterator(int num_v, int64_t n_edges, std::vector<int> &offsets, std::vector<int> &csr, std::vector<int> &s_edge)
{
    cudaSetDevice(0);
    if (n_edges == 0)
        return 0;

    int *d_csr = nullptr, *d_ofs = nullptr, *d_res = nullptr, *d_s_edge = nullptr;
    unsigned long long *d_sum = nullptr;

    unsigned long long tri_count = 0;

    size_t s = n_edges * sizeof(int);
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    CHECK(cudaMallocAsync(&d_ofs, offsets.size() * sizeof(int), stream));
    CHECK(cudaMallocAsync(&d_csr, s, stream));
    CHECK(cudaMallocAsync(&d_s_edge, s, stream));
    CHECK(cudaMallocAsync(&d_res, s, stream));
    CHECK(cudaMallocAsync(&d_sum, sizeof(unsigned long long), stream));

    CHECK(cudaMemcpyAsync(d_ofs, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(d_csr, csr.data(), csr.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(d_s_edge, s_edge.data(), s_edge.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemsetAsync(d_sum, 0, sizeof(unsigned long long), stream));
    int64_t n_blocks = (n_edges + 127) / 128;
    dim3 blockDim(128);
    dim3 gridDim(n_blocks);
    cudaFuncSetCacheConfig(edge_search_tri, cudaFuncCachePreferL1);
    edge_search_tri<<<gridDim, blockDim, 0, stream>>>(num_v, n_edges, d_ofs, d_csr, d_s_edge, d_res);
    CHECK(cudaGetLastError());

    // Nessun sync necessario: kernel sullo stesso stream sono serializzati
    reduce_vector<<<gridDim, blockDim, blockDim.x * sizeof(unsigned long long), stream>>>(n_edges, d_res, d_sum);
    CHECK(cudaGetLastError());

    CHECK(cudaMemcpyAsync(&tri_count, d_sum, sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamSynchronize(stream));

    CHECK(cudaFreeAsync(d_res, stream));
    CHECK(cudaFreeAsync(d_csr, stream));
    CHECK(cudaFreeAsync(d_ofs, stream));
    CHECK(cudaFreeAsync(d_s_edge, stream));
    CHECK(cudaFreeAsync(d_sum, stream));
    CHECK(cudaStreamDestroy(stream));
    return (int64_t)tri_count;
}

out_type adaptive_edge_search(int num_v, int64_t n_edges, std::vector<int> &offsets, std::vector<int> &csr, std::vector<int> &s_edge, std::vector<int> &th_level, std::vector<int> &warp_level)
{

    cudaSetDevice(0);
    if (n_edges == 0)
        return 0;

    int *d_csr = nullptr, *d_ofs = nullptr, *d_res = nullptr, *d_s_edge = nullptr;
    unsigned long long *d_sum = nullptr;

    unsigned long long tri_count = 0;

    size_t s = n_edges * sizeof(int);
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    CHECK(cudaMallocAsync(&d_ofs, offsets.size() * sizeof(int), stream));
    CHECK(cudaMallocAsync(&d_csr, s, stream));
    CHECK(cudaMallocAsync(&d_s_edge, s, stream));
    CHECK(cudaMallocAsync(&d_res, s, stream));
    CHECK(cudaMallocAsync(&d_sum, sizeof(unsigned long long), stream));

    CHECK(cudaMemcpyAsync(d_ofs, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(d_csr, csr.data(), csr.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(d_s_edge, s_edge.data(), s_edge.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemsetAsync(d_sum, 0, sizeof(unsigned long long), stream));

    int th_level_size = th_level.size();
    int warp_level_size = warp_level.size();
    dim3 blkDim(128);
    dim3 grid_th_level((th_level_size + 127) / 128);
    dim3 grid_warp_level((warp_level_size + 3) / 4);
    dim3 grid_reduce((n_edges + 127) / 128);
    int *d_wp_level, *d_th_level;
    CHECK(cudaMallocAsync(&d_th_level, th_level_size * sizeof(int), stream));
    CHECK(cudaMallocAsync(&d_wp_level, warp_level_size * sizeof(int), stream));
    CHECK(cudaMemcpyAsync(d_th_level, th_level.data(), th_level_size * sizeof(int), cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(d_wp_level, warp_level.data(), warp_level_size * sizeof(int), cudaMemcpyHostToDevice, stream));

    cudaStream_t stream_th, stream_wp;
    cudaStreamCreate(&stream_th);
    cudaStreamCreate(&stream_wp);
    cudaEvent_t data_ready;
    CHECK(cudaEventCreate(&data_ready));
    CHECK(cudaEventRecord(data_ready, stream));
    CHECK(cudaStreamWaitEvent(stream_th, data_ready, 0));
    CHECK(cudaStreamWaitEvent(stream_wp, data_ready, 0));
    if (th_level_size > 0)
        edge_thread_search_tri<<<grid_th_level, blkDim, 0, stream_th>>>(num_v, th_level_size, d_ofs, d_csr, d_s_edge, d_res, d_th_level);
    if (warp_level_size > 0)
        edge_warp_search_tri<<<grid_warp_level, blkDim, 0, stream_wp>>>(num_v, warp_level_size, d_ofs, d_csr, d_s_edge, d_res, d_wp_level);
    CHECK(cudaGetLastError());
    // sync streams
    cudaEvent_t th_done, wp_done;
    CHECK(cudaEventCreate(&th_done));
    CHECK(cudaEventCreate(&wp_done));
    CHECK(cudaEventRecord(th_done, stream_th));
    CHECK(cudaEventRecord(wp_done, stream_wp));
    CHECK(cudaStreamWaitEvent(stream, th_done, 0));
    CHECK(cudaStreamWaitEvent(stream, wp_done, 0));

    reduce_vector<<<grid_reduce, blkDim, blkDim.x * sizeof(unsigned long long), stream>>>(n_edges, d_res, d_sum);
    CHECK(cudaGetLastError());

    CHECK(cudaEventDestroy(data_ready));
    CHECK(cudaEventDestroy(th_done));
    CHECK(cudaEventDestroy(wp_done));
    CHECK(cudaStreamDestroy(stream_th));
    CHECK(cudaStreamDestroy(stream_wp));

    CHECK(cudaFreeAsync(d_th_level, stream));
    CHECK(cudaFreeAsync(d_wp_level, stream));
    CHECK(cudaMemcpyAsync(&tri_count, d_sum, sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamSynchronize(stream));

    CHECK(cudaFreeAsync(d_res, stream));
    CHECK(cudaFreeAsync(d_csr, stream));
    CHECK(cudaFreeAsync(d_ofs, stream));
    CHECK(cudaFreeAsync(d_s_edge, stream));
    CHECK(cudaFreeAsync(d_sum, stream));
    CHECK(cudaStreamDestroy(stream));
    return (int64_t)tri_count;
}



out_type adaptive_help_search(int num_v, int64_t n_edges, std::vector<int> &offsets, std::vector<int> &csr, std::vector<int> &s_edge, std::vector<int> &th_level, std::vector<int> &warp_level)
{

    cudaSetDevice(0);
    if (n_edges == 0)
        return 0;

    int *d_csr = nullptr, *d_ofs = nullptr, *d_res = nullptr, *d_s_edge = nullptr;
    unsigned long long *d_sum = nullptr;

    unsigned long long tri_count = 0;

    size_t s = n_edges * sizeof(int);
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    CHECK(cudaMallocAsync(&d_ofs, offsets.size() * sizeof(int), stream));
    CHECK(cudaMallocAsync(&d_csr, s, stream));
    CHECK(cudaMallocAsync(&d_s_edge, s, stream));
    CHECK(cudaMallocAsync(&d_res, s, stream));
    CHECK(cudaMallocAsync(&d_sum, sizeof(unsigned long long), stream));

    CHECK(cudaMemcpyAsync(d_ofs, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(d_csr, csr.data(), csr.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(d_s_edge, s_edge.data(), s_edge.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemsetAsync(d_sum, 0, sizeof(unsigned long long), stream));

    int th_level_size = th_level.size();
    int warp_level_size = warp_level.size();
    dim3 blkDim(128);
    dim3 grid_th_level((th_level_size + 127) / 128);
    dim3 grid_warp_level((warp_level_size + 3) / 4);
    dim3 grid_reduce((n_edges + 127) / 128);
    int *d_wp_level, *d_th_level;
    CHECK(cudaMallocAsync(&d_th_level, th_level_size * sizeof(int), stream));
    CHECK(cudaMallocAsync(&d_wp_level, warp_level_size * sizeof(int), stream));
    CHECK(cudaMemcpyAsync(d_th_level, th_level.data(), th_level_size * sizeof(int), cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(d_wp_level, warp_level.data(), warp_level_size * sizeof(int), cudaMemcpyHostToDevice, stream));

    cudaStream_t stream_th, stream_wp;
    cudaStreamCreate(&stream_th);
    cudaStreamCreate(&stream_wp);
    cudaEvent_t data_ready;
    CHECK(cudaEventCreate(&data_ready));
    CHECK(cudaEventRecord(data_ready, stream));
    CHECK(cudaStreamWaitEvent(stream_th, data_ready, 0));
    CHECK(cudaStreamWaitEvent(stream_wp, data_ready, 0));
    if (th_level_size > 0)
        edge_thread_search_tri<<<grid_th_level, blkDim, 0, stream_th>>>(num_v, th_level_size, d_ofs, d_csr, d_s_edge, d_res, d_th_level);
    if (warp_level_size > 0)
        edge_warp_search_tri<<<grid_warp_level, blkDim, 0, stream_wp>>>(num_v, warp_level_size, d_ofs, d_csr, d_s_edge, d_res, d_wp_level);
    CHECK(cudaGetLastError());
    // sync streams
    cudaEvent_t th_done, wp_done;
    CHECK(cudaEventCreate(&th_done));
    CHECK(cudaEventCreate(&wp_done));
    CHECK(cudaEventRecord(th_done, stream_th));
    CHECK(cudaEventRecord(wp_done, stream_wp));
    CHECK(cudaStreamWaitEvent(stream, th_done, 0));
    CHECK(cudaStreamWaitEvent(stream, wp_done, 0));

    reduce_vector<<<grid_reduce, blkDim, blkDim.x * sizeof(unsigned long long), stream>>>(n_edges, d_res, d_sum);
    CHECK(cudaGetLastError());

    CHECK(cudaEventDestroy(data_ready));
    CHECK(cudaEventDestroy(th_done));
    CHECK(cudaEventDestroy(wp_done));
    CHECK(cudaStreamDestroy(stream_th));
    CHECK(cudaStreamDestroy(stream_wp));

    CHECK(cudaFreeAsync(d_th_level, stream));
    CHECK(cudaFreeAsync(d_wp_level, stream));
    CHECK(cudaMemcpyAsync(&tri_count, d_sum, sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamSynchronize(stream));

    CHECK(cudaFreeAsync(d_res, stream));
    CHECK(cudaFreeAsync(d_csr, stream));
    CHECK(cudaFreeAsync(d_ofs, stream));
    CHECK(cudaFreeAsync(d_s_edge, stream));
    CHECK(cudaFreeAsync(d_sum, stream));
    CHECK(cudaStreamDestroy(stream));
    return (int64_t)tri_count;
}
