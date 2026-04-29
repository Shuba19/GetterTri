#ifndef TENSORUTILITIES
#define TENSORUTILITIES
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <cstdint>


#define TILE_SIDE 16
#define TILE_BUILDER_THREADS 128
#define TILE_ROWS_PER_GROUP 16
#define TILE_GROUPS_PER_BLOCK (TILE_BUILDER_THREADS / TILE_ROWS_PER_GROUP)
struct tiles
{
    double tile[256];
};

struct tiles_b
{
    u_int16_t tile[16] = {0};
};

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
__global__ void tiles_builder(int tpr, int num_v, int total_t, const int *__restrict__ csr, const int *__restrict__ ofs, tiles_b *__restrict__ matrix);
__device__ __forceinline__ int triangular_col_from_id(int id)
{
    int col = 0;
    while ((col * (col + 1)) / 2 <= id)
        ++col;
    return col - 1;
}

__device__ __forceinline__ int from_x_y_to_id(int x, int y)
{
    return x * (x + 1) / 2 + y;
}
//DEBUG
__device__ void print_tile(tiles_b tile);
__device__ void print_tile_int(int *tile);


#endif