#include "../../../../Libs/ChronoCuda.h"
#include "../../../../Libs/GraphReader.h"
#include "../../../../Libs/CudaUtilities.h"
#include "../../../../Libs/TensorUtilities.h"
#include "../../../../Libs/TileReader.h"
#include <chrono>
#include <vector>
#include <iostream>
#include <fstream>

using namespace nvcuda;

#define BATCHSIZE (5 << 25)

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

/*
QUESTO FILE SERVE PER LA CONVERSIONE DEI GRAFI IN SPARSE TILE.
Vengono generati BATCHSIZE tile alla volta, e ogni tile è rappresentato da 16 bit (per una tile di lato 16) che indicano la presenza o assenza di un arco. Il processo è il seguente:
1. Si calcola il numero totale di tile necessarie per coprire la matrice
*/

__global__ void tiles_builder(int tpr, int num_v, uint64_t total_t, uint64_t batch_size, const int *__restrict__ csr, const int *__restrict__ ofs, const uint64_t id, tiles_b *__restrict__ matrix);
TILES calculate_valid_tiles(GraphData g);
void create_stile_file(TILES t, const std::string &filename = "stile.bin");

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <graph_file>" << std::endl;
        return 1;
    }

    GraphData g = readGraph(argv[1]);
    TILES t = calculate_valid_tiles(g);
    std::string output_file = "stile.bin";
    if (argc >= 3)
    {
        output_file = argv[2];
    }
    create_stile_file(t, output_file);
    return 0;
}

void print_tile_host(tiles_b tile)
{
    for (int i = 0; i < TILE_SIDE; i++)
    {
        for (int j = 0; j < TILE_SIDE; j++)
        {
            std::cout << ((tile.tile[i] >> (TILE_SIDE - 1 - j)) & 1) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "-----------------" << std::endl;
}

TILES calculate_valid_tiles(GraphData g)
{
    const int num_v = g.num_v;
    const uint64_t tpr = (num_v + 15) >> 4;
    const uint64_t total_t = tpr * (tpr + 1) / 2;
    std::cout << "Calculating valid tiles for graph with " << num_v << " vertices and " << g.num_edge << " edges." << std::endl;
    std::cout << "Total tiles to process: " << total_t << std::endl;
    std::cout << "Tiles per row/column: " << tpr << std::endl;

    int *d_csr, *d_ofs;
    tiles_b *d_matrix;
    TILES t;
    t.num_v = num_v;
    t.num_e = g.num_edge;
    cudaSetDevice(0);
    CHECK(cudaMalloc(&d_csr, g.csr.size() * sizeof(int)));
    CHECK(cudaMalloc(&d_ofs, g.offsets.size() * sizeof(int)));
    CHECK(cudaMemcpy(d_csr, g.csr.data(), g.csr.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_ofs, g.offsets.data(), g.offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(&d_matrix, BATCHSIZE * sizeof(tiles_b)));

    std::cout << "BATCH SIZE: " << BATCHSIZE << std::endl;
    std::cout << "Total tiles: " << total_t << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<tiles_b> temp(BATCHSIZE);

    for (uint64_t id = 0; id < total_t; id += BATCHSIZE)
    {
        const uint64_t batch_size = (id + BATCHSIZE > total_t) ? (total_t - id) : (uint64_t)BATCHSIZE;

        const int num_blocks = (int)((batch_size + TILE_GROUPS_PER_BLOCK - 1) / TILE_GROUPS_PER_BLOCK);

        tiles_builder<<<num_blocks, TILE_GROUPS_PER_BLOCK * TILE_ROWS_PER_GROUP>>>(tpr, num_v, total_t, batch_size, d_csr, d_ofs, id, d_matrix);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(temp.data(), d_matrix, batch_size * sizeof(tiles_b), cudaMemcpyDeviceToHost));

        std::cout << "Processed tiles: " << id + batch_size << " / " << total_t << std::endl;

        for (uint64_t i = 0; i < batch_size; i++)
        {
            uint16_t s = 0;
            for (int j = 0; j < TILE_SIDE; j++)
            {
                s |= temp[i].tile[j];
            }
            if (s == 0)
                continue;

            const uint64_t global_id = id + i;
            if (global_id < total_t)
            {
                t.tiles.push_back(temp[i]);
                t.tile_ids.push_back(global_id);
            }
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(t1 - t0);
    std::cout << "Tile conversion time: " << duration.count() << " s" << std::endl;

    cudaFree(d_csr);
    cudaFree(d_ofs);
    cudaFree(d_matrix);

    return t;
}
void create_stile_file(TILES t, const std::string &filename)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "Errore creazione file binario" << std::endl;
        return;
    }

    // 1. Scrivi header
    file.write(reinterpret_cast<const char *>(&t.num_v), sizeof(int));
    file.write(reinterpret_cast<const char *>(&t.num_e), sizeof(int));

    uint64_t num_valid_tiles = t.tiles.size();
    file.write(reinterpret_cast<const char *>(&num_valid_tiles), sizeof(uint64_t));

    std::cout << "Number of vertices: " << t.num_v << std::endl;
    std::cout << "Number of edges: " << t.num_e << std::endl;
    std::cout << "Number of valid tiles: " << num_valid_tiles << std::endl;

    for (size_t i = 0; i < num_valid_tiles; ++i)
    {
        file.write(reinterpret_cast<const char *>(&t.tile_ids[i]), sizeof(uint64_t));
        file.write(reinterpret_cast<const char *>(t.tiles[i].tile), 16 * sizeof(u_int16_t));
    }

    file.close();
}
__global__ void tiles_builder(int tpr, int num_v, uint64_t total_t, uint64_t batch_size, const int *__restrict__ csr, const int *__restrict__ ofs, const uint64_t id, tiles_b *__restrict__ matrix)
{
    (void)tpr;

    const uint64_t global_thread = (uint64_t)blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t tile_id = global_thread / TILE_ROWS_PER_GROUP + id;
    const int local_row = threadIdx.x & (TILE_ROWS_PER_GROUP - 1);
    const int tile_slot = threadIdx.x / TILE_ROWS_PER_GROUP;

    __shared__ int shared_tile_col[TILE_GROUPS_PER_BLOCK];
    __shared__ int shared_tile_row[TILE_GROUPS_PER_BLOCK];

    if (local_row == 0 && tile_id < total_t && (tile_id - id) < batch_size)
    {
        const int tile_col = triangular_col_from_id(tile_id);
        shared_tile_col[tile_slot] = tile_col;
        shared_tile_row[tile_slot] = (int)(tile_id - (uint64_t)tile_col * (tile_col + 1) / 2);
    }
    __syncthreads();
    if (tile_id >= total_t || (tile_id - id) >= batch_size)
        return;

    const int tile_col = shared_tile_col[tile_slot];
    const int tile_row = shared_tile_row[tile_slot];
    const int start_x = tile_col * TILE_SIDE;
    const int y = tile_row * TILE_SIDE + local_row;

    if (y >= num_v || start_x >= num_v)
    {
        matrix[tile_id - id].tile[local_row] = 0;
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

    matrix[tile_id - id].tile[local_row] = row_bits;
}