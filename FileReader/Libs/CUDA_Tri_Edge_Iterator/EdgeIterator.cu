#include "../CommonMethods/common_methods.h"
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


__global__ void edge_search_tri(int num_v, int64_t num_e, int *ofs, int *csr, int *results)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < num_e)
    {
        int n_tri = 0;
        int s_node = searchSourceNode(ofs, num_v, id) - 1;
        int d_node = csr[id];
        if (d_node <= s_node)
        {
            results[id] = 0;
            return;
        }
        int s1 = ofs[s_node], e1 = ofs[s_node + 1];
        int s2 = ofs[d_node], e2 = ofs[d_node + 1];
        while (s1 < e1 && s2 < e2)
        {
            int c1 = csr[s1], c2 = csr[s2];
            if (c1 == c2)
            {
                if (c1 > d_node)
                {
                    n_tri++;
                }
                s1++;
                s2++;
            }
            else if (c1 < c2)
                s1++;
            else
                s2++;
        }
        results[id] = n_tri;
        return;
    }
}
__global__ void edge_search_tri_directed(int num_v, int64_t num_e, int *ofs, int *csr, int *results)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_e)
        return;

    int n_tri = 0;

    int s_node = searchSourceNode(ofs, num_v, id) - 1;
    int d_node = csr[id];

    int s1 = ofs[s_node];
    int e1 = ofs[s_node + 1];
    int s2 = ofs[d_node];
    int e2 = ofs[d_node + 1];

    while (s1 < e1 && s2 < e2)
    {
        int c1 = csr[s1];
        int c2 = csr[s2];

        if (c1 == c2)
        {
            if (c1 > d_node)
                n_tri++;
            s1++;
            s2++;
        }
        else if (c1 < c2)
        {
            s1++;
        }
        else
        {
            s2++;
        }
    }
    results[id] = n_tri;
}


out_type SearchTriangle_Edge_Iterator(int num_v, int64_t n_edges, std::vector<int> &offsets, std::vector<int> &csr, bool undirected)
{
    cudaSetDevice(0);
    int *d_csr = nullptr, *d_ofs = nullptr, *d_res = nullptr;
    unsigned long long *d_sum = nullptr;
    n_edges = n_edges<<1;
    size_t s = n_edges *sizeof(int);
    int64_t n_blocks = (n_edges + 127) / 128;
    CHECK(cudaMalloc(&d_ofs, (offsets.size()) * sizeof(int)));
    CHECK(cudaMalloc(&d_csr, s));
    CHECK(cudaMalloc(&d_res, s));
    CHECK(cudaMalloc(&d_sum, sizeof(unsigned long long)));
    CHECK(cudaMemcpy(d_ofs, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_csr, csr.data(),csr.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_sum, 0, sizeof(unsigned long long)));
    dim3 blockDim(128);
    dim3 gridDim(n_blocks);
    if (undirected)
        edge_search_tri<<<gridDim, blockDim>>>(num_v, n_edges, d_ofs, d_csr, d_res);
    else
        edge_search_tri_directed<<<gridDim, blockDim>>>(num_v, n_edges, d_ofs, d_csr, d_res);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    reduce_vector<<<gridDim, blockDim, blockDim.x * sizeof(unsigned long long)>>>(n_edges, d_res, d_sum);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    unsigned long long tri_count = 0;
    CHECK(cudaMemcpy(&tri_count, d_sum, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    cudaFree(d_res);
    cudaFree(d_csr);
    cudaFree(d_ofs);
    cudaFree(d_sum);

    int64_t n_tri = (int64_t)tri_count;
    if (!undirected) n_tri /= 3;

    return n_tri;
}
