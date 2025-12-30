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
                count += bin_search(pivot, &csr[of1], of2 - of1) ? 1 : 0;
            }
        }
        results[id] = count;
    }
}

__global__ void static d_search_tri_directed(int num_v, int *ofs, int *csr, int *results)
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
            for (int j = ofs[index]; j < ofs[index + 1]; j++)
            {
                int pivot = csr[j];
                count += bin_search(pivot, &csr[of1], of2 - of1) ? 1 : 0;
            }
        }
        results[id] = count;
    }
}

out_type SearchTriangle_Node_Iterator(int num_v, int64_t n_edges, std::vector<int> &offsets, std::vector<int> &csr, bool undirect)
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

    if (undirect)
        d_search_tri<<<gridDim, blockDim>>>(num_v, d_ofs, d_csr, d_res);
    else
        d_search_tri_directed<<<gridDim, blockDim>>>(num_v, d_ofs, d_csr, d_res);
    cudaDeviceSynchronize();
    cudaFree(d_csr);
    cudaFree(d_ofs);
    int64_t n_tri = 0;
    int64_t nr_blocks = (num_v + 127) / 128; 
    dim3 r_blockDim(128);
    dim3 r_gridDim(nr_blocks);
    unsigned long long *d_sum = nullptr;

    CHECK(cudaMalloc(&d_sum, sizeof(unsigned long long)));
    CHECK(cudaMemset(d_sum, 0, sizeof(unsigned long long))); 
    
    reduce_vector<<<r_gridDim, r_blockDim>>>(num_v, d_res, d_sum);
    cudaDeviceSynchronize();
    unsigned long long h_sum = 0;
    CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    n_tri = h_sum;
    CHECK(cudaFree(d_sum));
    CHECK(cudaFree(d_res));
    if(undirect)
        n_tri = n_tri;
    return n_tri;
}