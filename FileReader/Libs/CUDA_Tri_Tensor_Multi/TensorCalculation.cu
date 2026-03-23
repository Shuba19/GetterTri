#include "../CommonMethods/common_methods.h"
using namespace nvcuda;

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



__global__ void countTriangle(int tpr, tiles_b *matrix, double *square)
{
    int tile_id = blockIdx.x;
    int row = threadIdx.y;
    int col = threadIdx.x;
    int tid = row * 16 + col;

    __shared__ half A[256];
    __shared__ half B[256];
    __shared__ double C[256];
    __shared__ float temp_C[256];

    int t_col = triangular_col_from_id(tile_id);
    int t_row = tile_id - t_col * (t_col + 1) / 2;

    C[tid] = 0.0;
    temp_C[tid] = 0.0f;
    __syncthreads();
#pragma unroll
    for (int k_tile = 0; k_tile < tpr; k_tile++)
    {
        const int r2 = min(t_row, k_tile);
        const int c2 = max(t_row, k_tile);
        const int id2 = c2 * (c2 + 1) / 2 + r2;

        // tile che contiene A[k, t_col] → indici (min(k,t_col), max(k,t_col))
        const int r1 = min(k_tile, t_col);
        const int c1 = max(k_tile, t_col);
        const int id1 = c1 * (c1 + 1) / 2 + r1;

        if (t_col > k_tile)
        {
            u_int16_t a_val = matrix[id1].tile[col];
            A[tid] = __int2half_ru((a_val >> (15 - row)) & 1);
        }
        else
        {
            u_int16_t a_row = matrix[id1].tile[row];
            A[tid] = __int2half_ru((a_row >> (15 - col)) & 1);
        }

        if (k_tile > t_row)
        {
            u_int16_t b_val = matrix[id2].tile[col];
            B[tid] = __int2half_ru((b_val >> (15 - row)) & 1);
        }
        else
        {
            u_int16_t b_row = matrix[id2].tile[row];
            B[tid] = __int2half_ru((b_row >> (15 - col)) & 1);
        }
        __syncthreads();
        if (tid < 32)
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

            wmma::load_matrix_sync(a_frag, A, 16);
            wmma::load_matrix_sync(b_frag, B, 16);
            wmma::fill_fragment(c_frag, 0.0f);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            wmma::store_matrix_sync(temp_C, c_frag, 16, wmma::mem_row_major);
        }
        __syncthreads();
        C[tid] += (double)temp_C[tid];
        __syncthreads();
    }
    int64_t tile_offset = (int64_t)tile_id * 256;
    square[tile_offset + tid] = C[tid];
}

__global__ void cubeMatrix(int tpr, tiles_b *matrix, double *square, int64_t *diag, int num_v)
{
    int tile_id = blockIdx.x;
    int row = threadIdx.y;
    int col = threadIdx.x;
    int tid = row * 16 + col;

    __shared__ half A[256];
    __shared__ half B[256];
    __shared__ double C[256];
    __shared__ float temp_C[256];

    int t_col = triangular_col_from_id(tile_id);
    int t_row = tile_id - t_col * (t_col + 1) / 2;

    C[tid] = 0.0;
    temp_C[tid] = 0.0f;
    __syncthreads();
#pragma unroll
    for (int k_tile = 0; k_tile < tpr; k_tile++)
    {
        int r1 = max(t_col, k_tile);
        int c1 = min(t_col, k_tile);
        int id1 = r1 * (r1 + 1) / 2 + c1;

        int r2 = max(k_tile, t_row);
        int c2 = min(k_tile, t_row);
        int id2 = r2 * (r2 + 1) / 2 + c2;
        if (t_col < k_tile)
        {
            double a = square[(int64_t)id1 * 256 + col * 16 + row];
            A[tid] = __double2half(a);
        }
        else
        {
            double a = square[(int64_t)id1 * 256 + tid];
            A[tid] = __double2half(a);
        }

        if (k_tile > t_row)
        {
            u_int16_t b_val = matrix[id2].tile[col];
            B[tid] = __int2half_ru((b_val >> (15 - row)) & 1);
        }
        else
        {
            u_int16_t b_row_val = matrix[id2].tile[row];
            B[tid] = __int2half_ru((b_row_val >> (15 - col)) & 1);
        }
        __syncthreads();
        if (tid < 32)
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

            wmma::load_matrix_sync(a_frag, A, 16);
            wmma::load_matrix_sync(b_frag, B, 16);
            wmma::fill_fragment(c_frag, 0.0f);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            wmma::store_matrix_sync(temp_C, c_frag, 16, wmma::mem_row_major);
        }
        __syncthreads();
        C[tid] += (double)temp_C[tid];
        __syncthreads();
    }

    if (t_col == t_row)
    {
        if (row == col)
        {
            int global_diag_idx = t_col * 16 + row;

            if (global_diag_idx < num_v)
            {
                diag[global_diag_idx] = (int)C[tid];
            }
        }
    }
}

out_type TTC(int num_v, int64_t n_edges, std::vector<int> offsets, std::vector<int> csr)
{
    cudaSetDevice(0);
    int tiles_per_row = ((num_v + 15) >> 4);
    int64_t total_tiles = tiles_per_row * (tiles_per_row + 1) >> 1;
    int padded_size_csr = ((n_edges + 15) >> 4) << 4;
    int *d_csr, *d_ofs;
    tiles_b *d_tiles;
    d_csr = nullptr;
    d_ofs = nullptr;
    CHECK(cudaMalloc(&d_csr, (padded_size_csr) * sizeof(int)));
    CHECK(cudaMalloc(&d_ofs, (num_v + 1) * sizeof(int)));
    int tiles_shifted = total_tiles;
    CHECK(cudaMalloc(&d_tiles, (tiles_shifted) * sizeof(tiles_b)));
    CHECK(cudaMemcpy(d_csr, csr.data(), n_edges * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_ofs, offsets.data(), (num_v + 1) * sizeof(int), cudaMemcpyHostToDevice));
    dim3 tb_dim_grid((total_tiles + TILE_GROUPS_PER_BLOCK - 1) / TILE_GROUPS_PER_BLOCK);
    chrono_cuda timer("TTC");
    timer.cc_start();
    tiles_builder<<<tb_dim_grid, TILE_BUILDER_THREADS>>>(tiles_per_row, num_v, total_tiles, d_csr, d_ofs, d_tiles);
    cudaFree(d_csr);
    cudaFree(d_ofs);
    int64_t n_v = total_tiles * 256;
    double *d_square;
    CHECK(cudaMalloc(&d_square, (n_v) * sizeof(double)));
    dim3 blocks_dimension(16, 16);
    dim3 grid_dimension(total_tiles);
    int64_t *d_diag;
    CHECK(cudaMalloc(&d_diag, num_v * sizeof(int64_t)));
    CHECK(cudaMemset(d_diag, 0, num_v * sizeof(int64_t)));

    CHECK(cudaGetLastError());
    countTriangle<<<total_tiles, blocks_dimension>>>(tiles_per_row, d_tiles, d_square);

    cubeMatrix<<<total_tiles, blocks_dimension>>>(tiles_per_row, d_tiles, d_square, d_diag, num_v);
    out_type n_tri = 0;

    int64_t nr_blocks = (num_v + 127) / 128;
    dim3 r_blockDim(128);
    dim3 r_gridDim(nr_blocks);
    unsigned long long *d_sum = nullptr;

    CHECK(cudaMalloc(&d_sum, sizeof(unsigned long long)));
    CHECK(cudaMemset(d_sum, 0, sizeof(unsigned long long)));

    cudaFree(d_tiles);
    cudaFree(d_square);

    reduce_vector<<<r_gridDim, r_blockDim>>>(num_v, d_diag, d_sum);
    cudaDeviceSynchronize();
    timer.cc_stop();
    unsigned long long h_sum = 0;
    CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    n_tri = h_sum;
    cudaFree(d_diag);
    return n_tri / 6;
}
