#include "../CommonMethods/common_methods.h"
using namespace nvcuda;
#define TILE_SIDE 16
#define TILE_BUILDER_THREADS 32
#define TILE_ROWS_PER_GROUP 16
#define TILE_GROUPS_PER_BLOCK (TILE_BUILDER_THREADS / TILE_ROWS_PER_GROUP)

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

__device__ __forceinline__ int lower_bound_device(const int *__restrict__ values, int begin, int end, int key)
{
    int left = begin;
    int right = end;
    while (left < right)
    {
        int mid = left + ((right - left) >> 1);
        if (values[mid] < key)
            left = mid + 1;
        else
            right = mid;
    }
    return left;
}

__device__ int triangular_col_from_id(int id)
{
    int col = 0;
    while ((col * (col + 1)) / 2 <= id)
        ++col;
    return col - 1;
}

__global__ void tiles_builder(int tpr, int num_v, int total_t, const int *__restrict__ csr, const int *__restrict__ ofs, tiles_b *__restrict__ matrix)
{
    (void)tpr;

    const int global_thread = blockDim.x * blockIdx.x + threadIdx.x;
    const int tile_id = global_thread / TILE_ROWS_PER_GROUP;
    const int local_row = threadIdx.x & (TILE_ROWS_PER_GROUP - 1);
    const int tile_slot = threadIdx.x / TILE_ROWS_PER_GROUP;

    __shared__ int shared_tile_col[TILE_GROUPS_PER_BLOCK];
    __shared__ int shared_tile_row[TILE_GROUPS_PER_BLOCK];

    if (local_row == 0 && tile_id < total_t)
    {
        const int tile_col = triangular_col_from_id(tile_id);
        shared_tile_col[tile_slot] = tile_col;
        shared_tile_row[tile_slot] = tile_id - tile_col * (tile_col + 1) / 2;
    }
    __syncthreads();

    if (tile_id >= total_t)
        return;

    const int tile_col = shared_tile_col[tile_slot];
    const int tile_row = shared_tile_row[tile_slot];
    const int start_x = tile_col * TILE_SIDE;
    const int y = tile_row * TILE_SIDE + local_row;

    if (y >= num_v || start_x >= num_v)
    {
        matrix[tile_id].tile[local_row] = 0;
        return;
    }

    const int row_begin = ofs[y];
    const int row_end = ofs[y + 1];
    const int col_limit = min(start_x + TILE_SIDE, num_v);
    int pos = lower_bound_device(csr, row_begin, row_end, start_x);
    u_int16_t row_bits = 0;

    while (pos < row_end)
    {
        const int x = csr[pos];
        if (x >= col_limit)
            break;

        row_bits |= static_cast<u_int16_t>(1u << (TILE_SIDE - 1 - (x - start_x)));
        ++pos;
    }

    matrix[tile_id].tile[local_row] = row_bits;
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

// versione con 256 th lanciati
__global__ void square_then_hadamard(int tpr, tiles_b *matrix, unsigned long long *res, int num_v)
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

// versione che usa 32 thread per blocco

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
    for (int k_tile = 0; k_tile < tpr; k_tile++)
    {
        const int r1 = max(t_col, k_tile);
        const int c1 = min(t_col, k_tile);
        const int id1 = r1 * (r1 + 1) / 2 + c1;
        const int r2 = max(k_tile, t_row);
        const int c2 = min(k_tile, t_row);
        const int id2 = r2 * (r2 + 1) / 2 + c2;
        for (int i = tid; i < 256; i += 32)
        {
            const int r = i / 16;
            const int c = i % 16;
            if (t_col > k_tile)
            {
                const uint16_t a_val = matrix[id1].tile[c];
                A[i] = ((a_val >> (15 - r)) & 1u);
            }
            else
            {
                const uint16_t a_val = matrix[id1].tile[r];
                A[i] = ((a_val >> (15 - c)) & 1u);
            }
            if (k_tile > t_row)
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
        __syncthreads();
        wmma::fragment<wmma::matrix_a, 16, 16, 16, unsigned char, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, unsigned char, wmma::row_major> b_frag;
        wmma::load_matrix_sync(a_frag, A, 16);
        wmma::load_matrix_sync(b_frag, B, 16);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    wmma::store_matrix_sync(temp_C, c_frag, 16, wmma::mem_row_major);
    __syncthreads();
    unsigned long long s = 0;
    int factor = (t_row == t_col) ? 1 : 2;
    __shared__ tiles_b tile_for_hadamard;
    if (tid == 0)
    {
        tile_for_hadamard = matrix[tile_id];
        // printf("Tile_id: %d, t_row: %d, t_col: %d, factor: %d\n", tile_id, t_row, t_col, factor);
    }
    __syncthreads();
    for (int i = 0; i < 256; i += 32)
    {
        const int idx = i + threadIdx.x;
        const int r = idx / 16;
        const int c = idx % 16;
        s += (unsigned long long)temp_C[idx] * ((tile_for_hadamard.tile[r] >> (15 - c)) & 1);
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

// lancio tpr * tpr blocchi
// la suddivisione è blockIdx.x -> refTile
//                   blockIdx.y -> y-esima moltiplicazione riferita all'xedsimo tile
// ogni blocco ha 32 thread, in teoria si prova a vedere se andando dritti all'

__device__ void print_tile(tiles_b tile)
{
    for (int i = 0; i < 16; i++)
    {
        uint16_t row = tile.tile[i];
        for (int j = 0; j < 16; j++)
        {
            printf("%d ", (row >> (15 - j)) & 1u);
        }
        printf("\n");
    }
}
__device__ void print_tile_int(int *tile)
{
    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            printf("%d ", tile[i * 16 + j]);
        }
        printf("\n");
    }
}

__global__ void square_then_hadamard_per_tile(int tpr, tiles_b *matrix, int *res, int num_v)
{
    (void)num_v;
    int tile_id = blockIdx.x;
    int k_tile = blockIdx.y;

    const int t_col = triangular_col_from_id(tile_id);
    const int t_row = tile_id - t_col * (t_col + 1) / 2;

    const int r1 = max(t_col, k_tile);
    const int c1 = min(t_col, k_tile);
    const int id1 = r1 * (r1 + 1) / 2 + c1;
    const int r2 = max(k_tile, t_row);
    const int c2 = min(k_tile, t_row);
    const int id2 = r2 * (r2 + 1) / 2 + c2;

    __shared__ uint8_t A[256];
    __shared__ uint8_t B[256];
    __shared__ int temp_C[256];
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> c_frag;
    wmma::fill_fragment(c_frag, 0);

    __shared__ tiles_b tile_a, tile_b;
    if (threadIdx.x == 0)
    {
        tile_a = matrix[id2];
        tile_b = matrix[id1];
    }
    __syncwarp();
    for (int i = threadIdx.x; i < 256; i += 32)
    {
        const int r = i / 16;
        const int c = i % 16;
        if (t_col < k_tile)
        {
            const uint16_t a_val = tile_a.tile[c];
            A[i] = ((a_val >> (15 - r)) & 1u);
        }
        else
        {
            const uint16_t a_val = tile_a.tile[r];
            A[i] = ((a_val >> (15 - c)) & 1u);
        }
        if (k_tile < t_row)
        {
            const uint16_t b_val = tile_b.tile[c];
            B[i] = ((b_val >> (15 - r)) & 1u);
        }
        else
        {
            const uint16_t b_val = tile_b.tile[r];
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
    int factor = (t_row == t_col) ? 1 : 2;
    __shared__ tiles_b tile_for_hadamard;
    if (threadIdx.x == 0)
    {
        tile_for_hadamard = matrix[tile_id];
    }
    __syncwarp();
    __syncwarp();
    int s = 0;
    for (int i = 0; i < 256; i += 32)
    {
        const int idx = i + threadIdx.x;
        const int r = idx / 16;
        const int c = idx % 16;
        s += temp_C[idx] * ((tile_for_hadamard.tile[r] >> (15 - c)) & 1);
    }
    s *= factor;
    atomicAdd(&res[tile_id], s);
    __syncwarp();
    if (blockIdx.y == 0 && threadIdx.x == 0)
    {
        printf("For tile :%d factor = %d, t_row = %d, t_col = %d\n", tile_id, factor, t_row, t_col);
    }
}

out_type TTC_3(int num_v, int64_t n_edges, std::vector<int> offsets, std::vector<int> csr)
{
    chrono_cuda data("TTC_3 Data"), tb("TTC_3 Tiles Builder"), timer("TTC_3 Counting"), timer2("TTC_3 Hadamard");
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
    square_then_hadamard_per_tile<<<grid_dimension, blocks_dimension>>>(tiles_per_row, d_tiles, d_res, num_v);
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
    CHECK(cudaMemcpy(&h_count, d_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    cudaFree(d_res);
    cudaFree(d_count);
    return h_count / 6;
}