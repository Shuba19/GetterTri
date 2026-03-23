#include "../CommonMethods/common_methods.h"
using namespace nvcuda;

// versione che usa 32 thread per blocco

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

///////// WIPI WIP WIP WIP
/*qui i tiles vengono generati al memoneto, non vengono salvati in memoria
** in generale ho provato a fare lanci, la combo 128 th per blocco, è la più funzionale
** quindi lancero 128 th per blocco, ogni blocco verrà partizionato in %32 e verrà assegnata ad un tile
** ogni thread della partizione ha questo compito cerco un valore della riga, esiste? si inserisco 1, altrimento 0
** dopo aver creato le due matrici A e B, faccio il prodotto matrice matrice con wmma, ogni thread calcola un elemento della matrice risultato,
** L'accumulator C, verrà salvato in shared memory, dopo si proverà con i registri per aumentare speed,
** questo processo verrà iterato fino a quando non si è ottenuto il quadrato del tile. a questo punto si creerà un tile per la costruzione del tile
** originale, e si applichera hadamard. Le ricerchè verranno eseguite tramite merge path
*/

__global__ void square_then_hadamard(int tpr, tiles_b  *__restrict__ matrix, unsigned long long *__restrict__ res, int num_v)
{
    int tile_id = blockIdx.x;
    int row = threadIdx.y;
    int col = threadIdx.x;
    int tid = row * 16 + col;

    // Tensor-core integer path: uint8 inputs with int accumulator.
    __shared__ uint8_t A[256];
    __shared__ uint8_t B[256];
    __shared__ int64_t C[256];
    __shared__ int temp_C[256];

    wmma::fragment<wmma::accumulator, 16, 16, 16, int> c_frag;
    int t_col = triangular_col_from_id(tile_id);
    int t_row = tile_id - t_col * (t_col + 1) / 2;

    C[tid] = 0;
    temp_C[tid] = 0;
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

        if (t_col > k_tile)
        {
            u_int16_t a_val = matrix[id1].tile[col];
            A[tid] = static_cast<uint8_t>((a_val >> (15 - row)) & 1u);
        }
        else
        {
            u_int16_t a_row = matrix[id1].tile[row];
            A[tid] = static_cast<uint8_t>((a_row >> (15 - col)) & 1u);
        }

        if (k_tile > t_row)
        {
            u_int16_t b_val = matrix[id2].tile[col];
            B[tid] = static_cast<uint8_t>((b_val >> (15 - row)) & 1u);
        }
        else
        {
            u_int16_t b_row = matrix[id2].tile[row];
            B[tid] = static_cast<uint8_t>((b_row >> (15 - col)) & 1u);
        }
        __syncthreads();
        if (tid < 32)
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, unsigned char, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, unsigned char, wmma::row_major> b_frag;
            wmma::load_matrix_sync(a_frag, A, 16);
            wmma::load_matrix_sync(b_frag, B, 16);
            wmma::fill_fragment(c_frag, 0);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            wmma::store_matrix_sync(temp_C, c_frag, 16, wmma::mem_row_major);
        }
        __syncthreads();
        C[tid] += static_cast<int64_t>(temp_C[tid]);
        __syncthreads();
    }
    __shared__ tiles_b tile_for_hadamard;
    if (tid == 0)
    {
        tile_for_hadamard = matrix[tile_id];
    }
    __syncthreads();
    // hadamard
    int val = tile_for_hadamard.tile[row] >> (15 - col) & 1;
    C[tid] = val * C[tid];
    __syncthreads();
    if (tid == 0)
    {
        unsigned long long count = 0;
        for (int i = 0; i < 256; i++)
        {
            // check if index is part of the diagonal or not
            //  if it is not, we need to mulitply by 2, because the triangle is counted twice
            int factor = (t_row == t_col) ? 1 : 2;
            count += static_cast<unsigned long long>(C[i]) * static_cast<unsigned long long>(factor);
        }
        atomicAdd(res, count);
    }
}

__global__ void square_then_hadamard_warped(int tpr, tiles_b *matrix, unsigned long long *res, int num_v)
{
    const int tile_id = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ uint8_t A[256];
    __shared__ uint8_t B[256];
    __shared__ int temp_C[256];
    const int t_col = triangular_col_from_id(tile_id);
    const int t_row = tile_id - t_col * (t_col + 1) / 2;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> c_frag;
    wmma::fill_fragment(c_frag, 0);
    int t1, t2, id1, id2;
#pragma unroll
    for (int k_tile = 0; k_tile < tpr; k_tile++)
    {

        id1 = (k_tile > t_col) ? from_x_y_to_id(k_tile, t_col) : from_x_y_to_id(t_col, k_tile);
        id2 = (t_row > k_tile) ? from_x_y_to_id(t_row, k_tile) : from_x_y_to_id(k_tile, t_row);
        t1 = (t_col > k_tile);
        t2 = (k_tile > t_row);
        #pragma unroll
        for (int i = tid; i < 256; i += 32)
        {
            const int r = i / 16;
            const int c = i % 16;
            if (t1)
            {
                const uint16_t a_val = matrix[id1].tile[c];
                A[i] = ((a_val >> (15 - r)) & 1u);
            }
            else
            {
                const uint16_t a_val = matrix[id1].tile[r];
                A[i] = ((a_val >> (15 - c)) & 1u);
            }
            if (t2)
            {
                const uint16_t b_val = matrix[id2].tile[c];
                B[i] = ((b_val >> (15 - r)) & 1u);
            }
            else
            {
                const uint16_t b_val = matrix[id2].tile[r];
                B[i] = ((b_val >> (15 - c)) & 1u);
            }
        }
        __syncwarp();
        wmma::fragment<wmma::matrix_a, 16, 16, 16, unsigned char, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, unsigned char, wmma::row_major> b_frag;
        wmma::load_matrix_sync(a_frag, A, 16);
        wmma::load_matrix_sync(b_frag, B, 16);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncwarp();
    }
    wmma::store_matrix_sync(temp_C, c_frag, 16, wmma::mem_row_major);
    __syncwarp();
    unsigned long long s = 0;
    int factor = (t_row != t_col) + 1; 
    __syncwarp();
    for (int i = 0; i < 256; i += 32)
    {
        const int idx = i + threadIdx.x;
        const int r = idx / 16;
        const int c = idx % 16;
        s += (unsigned long long)temp_C[idx] * ((matrix[tile_id].tile[c] >> (15 - r)) & 1);
    }
    s *= factor;
    atomicAdd(&res[tile_id], s);
}

out_type TTC_2(int num_v, int64_t n_edges, std::vector<int> offsets, std::vector<int> csr)
{
    cudaSetDevice(0);
    int64_t tiles_per_row = ((num_v + 15) >> 4);
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
    chrono_cuda timer("TTC_2");
    timer.cc_start();
    tiles_builder<<<tb_dim_grid, TILE_BUILDER_THREADS>>>(tiles_per_row, num_v, total_tiles, d_csr, d_ofs, d_tiles);

    dim3 grid_dimension(total_tiles);
    CHECK(cudaGetLastError());
    unsigned long long *d_res;
    CHECK(cudaMalloc(&d_res, total_tiles * sizeof(unsigned long long)));

    if (1)
    {
        dim3 blocks_dimension(32);
        cudaFuncSetCacheConfig(square_then_hadamard_warped, cudaFuncCachePreferL1);
        square_then_hadamard_warped<<<grid_dimension, blocks_dimension>>>(tiles_per_row, d_tiles, d_res, num_v);
    }
    else
    {
        dim3 blocks_dimension(16, 16);
        square_then_hadamard<<<grid_dimension, blocks_dimension>>>(tiles_per_row, d_tiles, d_res, num_v);
    }
    cudaFree(d_csr);
    cudaFree(d_ofs);

    unsigned long long *d_count = nullptr;
    CHECK(cudaMalloc(&d_count, sizeof(unsigned long long)));
    CHECK(cudaMemset(d_count, 0, sizeof(unsigned long long)));
    unsigned long long h_count = 0;
    dim3 r_blockDim(128);
    dim3 r_gridDim((total_tiles + 127) / 128);
    reduce_vector<<<r_gridDim, r_blockDim>>>(total_tiles, d_res, d_count);
    cudaDeviceSynchronize();
    timer.cc_stop();
    CHECK(cudaMemcpy(&h_count, d_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    cudaFree(d_tiles);
    cudaFree(d_count);
    return h_count / 6;
}
