#include "TensorUtilities.h"

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



//DEBUG
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
