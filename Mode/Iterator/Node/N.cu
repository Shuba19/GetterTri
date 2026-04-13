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
__global__ void static d_search_tri(int num_v, int *ofs, int *csr, int *results);
int SearchTriangle_Node_Iterator(int num_v, int64_t n_edges, std::vector<int> &offsets, std::vector<int> &csr);
int main(int argc, char *argv[])
{
    auto t1 = std::chrono::high_resolution_clock::now();
    if (argv[1] == nullptr)
    {
        std::cerr << "Error: No input file provided. Please provide a graph file as an argument." << std::endl;
        return EXIT_FAILURE;
    }
    GraphData graph_data = readGraph_Forward(argv[1]);
    auto t_read = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_read = t_read - t1;
    std::cout << "Graph reading time: " << elapsed_read.count() << " ms" << std::endl;
    int triangles = SearchTriangle_Node_Iterator(graph_data.num_v, graph_data.num_edge, graph_data.offsets, graph_data.csr);
    std::cout << "Number of triangles: " << triangles << std::endl;
    auto t2 = std::chrono::high_resolution_clock::now();
    // in milliseconds
    std::chrono::duration<double, std::milli> elapsed = t2 - t1;
    std::cout << "Execution time: " << elapsed.count() << " ms" << std::endl;
    return 0;
}

int SearchTriangle_Node_Iterator(int num_v, int64_t n_edges, std::vector<int> &offsets, std::vector<int> &csr)
{
    cudaSetDevice(0);
    dim3 blockDim(128);
    dim3 gridDim((num_v + 128 - 1) / 128);
    int64_t csr_size = csr.size();

    int *d_csr, *d_ofs, *d_res;
    d_csr = nullptr; 
    d_ofs = nullptr;
    d_res = nullptr;
    CHECK(cudaMalloc(&d_ofs, (offsets.size()) * sizeof(int)));
    CHECK(cudaMalloc(&d_csr, csr_size * sizeof(int)));
    CHECK(cudaMalloc(&d_res, num_v * sizeof(int)));

    CHECK(cudaMemcpy(d_ofs, offsets.data(), (num_v + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_csr, csr.data(), csr_size * sizeof(int), cudaMemcpyHostToDevice));
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    chrono_cuda timer("Node Iterator", stream);
    timer.cc_start();
    d_search_tri<<<gridDim, blockDim, 0, stream>>>(num_v, d_ofs, d_csr, d_res);
    cudaFree(d_csr);
    cudaFree(d_ofs);
    int64_t n_tri = 0;
    int64_t nr_blocks = (num_v + 127) / 128;
    dim3 r_blockDim(128);
    dim3 r_gridDim(nr_blocks);
    unsigned long long *d_sum = nullptr;

    CHECK(cudaMalloc(&d_sum, sizeof(unsigned long long)));
    CHECK(cudaMemset(d_sum, 0, sizeof(unsigned long long)));

    reduce_vector<<<r_gridDim, r_blockDim, 0, stream>>>(num_v, d_res, d_sum);
    cudaDeviceSynchronize();
    timer.cc_stop();
    unsigned long long h_sum = 0;
    CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    n_tri = h_sum;
    CHECK(cudaFree(d_sum));
    CHECK(cudaFree(d_res));
    return n_tri;
}

__global__ void static d_search_tri(int num_v, int *ofs, int *csr, int *results)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < num_v)
    {
        int of1, of2;
        of1 = ofs[id];
        of2 = ofs[id + 1];
        int count = 0;
        for (int i = of1; i < of2; i++)
        {
            int index = csr[i];
            if (index <= id)
                continue;
            for (int j = ofs[index]; j < ofs[index + 1]; j++)
            {
                int pivot = csr[j];
                if (pivot <= index)
                    continue;
                count += bin_search(csr, of1, of2, pivot);
            }
        }
        results[id] = count;
    }
}
