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

// lancio tpr * tpr blocchi
// la suddivisione è blockIdx.x -> refTile
//                   blockIdx.y -> y-esima moltiplicazione riferita all'xedsimo tile
// ogni blocco ha 32 thread, in teoria si prova a vedere se andando dritti all'

__global__ void tile_identification()
{

    int tile_id = blockIdx.x;
    int k_tile = blockIdx.y;
    const int t_col = triangular_col_from_id(tile_id);
    const int t_row = tile_id - t_col * (t_col + 1) / 2;
    int id1, id2;
    id1 = (k_tile > t_col) ? from_x_y_to_id(k_tile, t_col) : from_x_y_to_id(t_col, k_tile);
    id2 = (t_row > k_tile) ? from_x_y_to_id(t_row, k_tile) : from_x_y_to_id(k_tile, t_row);
    int t1, t2;
    t1 = (t_col > k_tile);
    t2 = (k_tile > t_row);
    printf("Tile id: %d, t_row: %d, t_col: %d, k_tile: %d, id1: %d, id2: %d, t1: %d, t2: %d \n", tile_id, t_row, t_col, k_tile, id1, id2, t1, t2);
}


__global__ void square_then_hadamard_per_tile(const tiles_b *__restrict__ matrix, int *__restrict__ res)
{
    int tile_id = blockIdx.x;
    int k_tile = blockIdx.y;
    const int t_col = triangular_col_from_id(tile_id);
    const int t_row = tile_id - t_col * (t_col + 1) / 2;
    int id1, id2;
    id1 = (k_tile > t_col) ? from_x_y_to_id(k_tile, t_col) : from_x_y_to_id(t_col, k_tile);
    id2 = (t_row > k_tile) ? from_x_y_to_id(t_row, k_tile) : from_x_y_to_id(k_tile, t_row);
    int t1, t2;
    t1 = (t_col > k_tile);
    t2 = (k_tile > t_row);
    __shared__ uint8_t A[256];
    __shared__ uint8_t B[256];
    __shared__ int temp_C[256];
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> c_frag;
    wmma::fill_fragment(c_frag, 0);
    for (int i = threadIdx.x; i < 256; i += 32)
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
    wmma::store_matrix_sync(temp_C, c_frag, 16, wmma::mem_row_major);
    __syncwarp();
    int factor = (t_row != t_col) + 1;
    int s = 0;
    #pragma unroll
    for (int i = 0; i < 256; i += 32)
    {
        const int idx = i + threadIdx.x;
        const int r = idx / 16;
        const int c = idx % 16;
        s += (unsigned long long)temp_C[idx] * ((matrix[tile_id].tile[c] >> (15 - r)) & 1);
    }
    s*= factor;
    atomicAdd(&res[tile_id], s);
}

out_type TTC_3(int num_v, int64_t n_edges, std::vector<int> offsets, std::vector<int> csr)
{
    chrono_cuda data("TTC_3 Data"), tb("TTC_3 Tiles Builder"), timer("TTC_3 Counting"), timer2("TTC_3 Hadamard"), timer3("TTC_3 Total");
    cudaSetDevice(0);
    int64_t tiles_per_row = ((num_v + 15) >> 4);
    int64_t total_tiles = tiles_per_row * (tiles_per_row + 1) >> 1;
    int padded_size_csr = ((n_edges + 15) >> 4) << 4;
    int *d_csr, *d_ofs;
    tiles_b *d_tiles;
    d_csr = nullptr;
    d_ofs = nullptr;
    data.cc_start();
    CHECK(cudaMalloc(&d_csr, (padded_size_csr) * sizeof(int)));
    CHECK(cudaMalloc(&d_ofs, (num_v + 1) * sizeof(int)));
    int tiles_shifted = total_tiles;
    CHECK(cudaMalloc(&d_tiles, (tiles_shifted) * sizeof(tiles_b)));
    CHECK(cudaMemcpy(d_csr, csr.data(), n_edges * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_ofs, offsets.data(), (num_v + 1) * sizeof(int), cudaMemcpyHostToDevice));
    data.cc_stop();
    dim3 tb_dim_grid((total_tiles + TILE_GROUPS_PER_BLOCK - 1) / TILE_GROUPS_PER_BLOCK);
    tb.cc_start();
    timer3.cc_start();
    tiles_builder<<<tb_dim_grid, TILE_BUILDER_THREADS>>>(tiles_per_row, num_v, total_tiles, d_csr, d_ofs, d_tiles);
    CHECK(cudaGetLastError());
    tb.cc_stop();
    cudaFree(d_csr);
    cudaFree(d_ofs);
    int *d_res;
    CHECK(cudaMalloc(&d_res, total_tiles * sizeof(int)));
    CHECK(cudaMemset(d_res, 0, total_tiles * sizeof(int)));
    timer2.cc_start();
    dim3 grid_dimension((unsigned int)total_tiles, (unsigned int)tiles_per_row);
    dim3 blocks_dimension(32);
    cudaFuncSetCacheConfig(square_then_hadamard_per_tile, cudaFuncCachePreferL1);
    square_then_hadamard_per_tile<<<grid_dimension, blocks_dimension>>>(d_tiles, d_res);
    CHECK(cudaGetLastError());
    timer2.cc_stop();
    cudaFree(d_tiles);
    unsigned long long *d_count = nullptr;
    CHECK(cudaMalloc(&d_count, sizeof(unsigned long long)));
    CHECK(cudaMemset(d_count, 0, sizeof(unsigned long long)));
    unsigned long long h_count = 0;
    dim3 r_blockDim(128);
    dim3 r_gridDim((total_tiles + 127) / 128);
    timer.cc_start();
    reduce_vector<<<r_gridDim, r_blockDim>>>(total_tiles, d_res, d_count);
    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
    timer.cc_stop();
    timer3.cc_stop();
    CHECK(cudaMemcpy(&h_count, d_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    cudaFree(d_res);
    cudaFree(d_count);
    return h_count / 6;
}
