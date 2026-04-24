#include "../../../../Libs/ChronoCuda.h"
#include "../../../../Libs/GraphReader.h"
#include "../../../../Libs/CudaUtilities.h"
#include "../../../../Libs/TensorUtilities.h"
#include "../../../../Libs/TileReader.h"
#include <chrono>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>

using namespace nvcuda;

// BATCHSIZE bilanciato per i 6GB della 4050 Mobile
#define BATCHSIZE (1 << 25) 

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

// Forward declarations
GraphData ReorderByDeg(const GraphData& input);
void create_stile_file(TILES t, const std::string &filename);

// Kernel ottimizzato per Ada Lovelace (RTX 4050)
__global__ void tiles_builder_optimized(
    int num_v, 
    uint64_t total_t, 
    uint64_t batch_size, 
    const int* __restrict__ csr, 
    const int* __restrict__ ofs, 
    const uint64_t base_id, 
    tiles_b* __restrict__ matrix)
{
    // Usiamo 16 thread per ogni tile. Ogni Warp (32 thread) processa 2 tile completi.
    const uint64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t local_tile_idx = thread_id >> 4; // thread_id / 16
    const int local_row = thread_id & 15;          // thread_id % 16

    if (local_tile_idx >= batch_size) return;

    const uint64_t global_tile_id = base_id + local_tile_idx;
    if (global_tile_id >= total_t) return;

    // Calcolo coordinate triangolari
    const int tile_col = triangular_col_from_id(global_tile_id);
    const int tile_row = (int)(global_tile_id - (uint64_t)tile_col * (tile_col + 1) / 2);
    
    const int start_x = tile_col << 4; // tile_col * 16
    const int y = (tile_row << 4) + local_row;

    uint16_t row_bits = 0;

    if (y < num_v && start_x < num_v)
    {
        // __ldg utilizza la cache Read-Only/L1 dedicata alle texture, eccellente per CSR
        const int row_begin = __ldg(&ofs[y]);
        const int row_end = __ldg(&ofs[y + 1]);
        
        int pos = lower_bound_device(csr, row_begin, row_end, start_x);
        const int col_limit = start_x + 16;

        // Loop di riempimento bitwise
        while (pos < row_end)
        {
            const int x = __ldg(&csr[pos]);
            if (x >= col_limit) break;
            row_bits |= (1u << (15 - (x - start_x)));
            pos++;
        }
    }

    matrix[local_tile_idx].tile[local_row] = row_bits;
}

TILES calculate_valid_tiles(GraphData g)
{
    const int num_v = g.num_v;
    const uint64_t tpr = (num_v + 15) >> 4;
    const uint64_t total_t = tpr * (tpr + 1) / 2;
    
    std::cout << "Vertices: " << num_v << " | Total Tiles: " << total_t << std::endl;
    
    int *d_csr, *d_ofs;
    tiles_b *unified_matrix; 
    TILES t;
    t.num_v = num_v;
    t.num_e = g.num_edge;
    
    // Allocazione statica per la GPU
    CHECK(cudaMalloc(&d_csr, g.csr.size() * sizeof(int)));
    CHECK(cudaMalloc(&d_ofs, (num_v + 1) * sizeof(int)));
    CHECK(cudaMemcpy(d_csr, g.csr.data(), g.csr.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_ofs, g.offsets.data(), (num_v + 1) * sizeof(int), cudaMemcpyHostToDevice));

    // Memoria Unificata per l'output (scrittura GPU -> lettura CPU)
    CHECK(cudaMallocManaged(&unified_matrix, BATCHSIZE * sizeof(tiles_b)));
    
    // Prefetch iniziale per la GPU
    cudaMemLocation loc = {cudaMemLocationTypeDevice, 0};
    cudaMemPrefetchAsync(d_csr, g.csr.size() * sizeof(int), loc,0);
    cudaMemPrefetchAsync(d_ofs, (num_v + 1) * sizeof(int), loc,0);

    auto t0 = std::chrono::high_resolution_clock::now();

    for (uint64_t id = 0; id < total_t; id += BATCHSIZE)
    {
        const uint64_t batch_size = std::min((uint64_t)BATCHSIZE, total_t - id);
        
        // 16 thread per tile, 256 thread per blocco (16 tile per blocco)
        const int threadsPerBlock = 256;
        const int num_blocks = (batch_size * 16 + threadsPerBlock - 1) / threadsPerBlock;

        tiles_builder_optimized<<<num_blocks, threadsPerBlock>>>(
            num_v, total_t, batch_size, d_csr, d_ofs, id, unified_matrix
        );

        CHECK(cudaDeviceSynchronize());

        // Filtraggio veloce CPU: usiamo uint64_t per controllare 4 righe alla volta (16 righe = 4 uint64)
        for (uint64_t i = 0; i < batch_size; i++)
        {
            const uint64_t* check = reinterpret_cast<const uint64_t*>(&unified_matrix[i]);
            if (check[0] == 0 && check[1] == 0 && check[2] == 0 && check[3] == 0) 
                continue;

            t.tiles.push_back(unified_matrix[i]);
            t.tile_ids.push_back(id + i);
        }
        std::cout << "Progress: " << (id + batch_size) * 100 / total_t << "%\r" << std::flush;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "\nConversion done in: " << std::chrono::duration_cast<std::chrono::seconds>(t1 - t0).count() << "s" << std::endl;

    cudaFree(d_csr);
    cudaFree(d_ofs);
    cudaFree(unified_matrix);

    return t;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <graph_file> [output_file]" << std::endl;
        return 1;
    }

    GraphData g = readGraph(argv[1]);
    std::cout << "Reordering graph by degree..." << std::endl;
    GraphData g_reordered = ReorderByDeg(g);
    
    TILES t = calculate_valid_tiles(g_reordered);
    
    std::string output_file = (argc >= 3) ? argv[2] : "stile.bin";
    create_stile_file(t, output_file);
    
    return 0;
}

// Reorder rimane uguale alla tua logica corretta
GraphData ReorderByDeg(const GraphData& input) {
    GraphData output;
    output.num_v = input.num_v;
    std::vector<std::pair<int, int>> deg_node(input.num_v);
    for (int i = 0; i < input.num_v; i++) 
        deg_node[i] = {input.offsets[i + 1] - input.offsets[i], i};

    std::sort(deg_node.begin(), deg_node.end());

    std::vector<int> old_to_new(input.num_v);
    for (int i = 0; i < input.num_v; i++) 
        old_to_new[deg_node[i].second] = i;

    output.offsets.assign(input.num_v + 1, 0);
    output.csr.reserve(input.csr.size());

    for (int i = 0; i < input.num_v; i++) {
        int old_id = deg_node[i].second;
        int start = input.offsets[old_id], end = input.offsets[old_id + 1];
        std::vector<int> neighbors;
        for (int j = start; j < end; j++) 
            neighbors.push_back(old_to_new[input.csr[j]]);
        
        std::sort(neighbors.begin(), neighbors.end());
        for (int n : neighbors) output.csr.push_back(n);
        output.offsets[i + 1] = (int)output.csr.size();
    }
    output.num_edge = (int)output.csr.size();
    return output;
}

void create_stile_file(TILES t, const std::string &filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) return;
    file.write((char*)&t.num_v, sizeof(int));
    file.write((char*)&t.num_e, sizeof(int));
    uint64_t nv = t.tiles.size();
    file.write((char*)&nv, sizeof(uint64_t));
    for (size_t i = 0; i < nv; ++i) {
        file.write((char*)&t.tile_ids[i], sizeof(uint64_t));
        file.write((char*)t.tiles[i].tile, 16 * sizeof(uint16_t));
    }
    file.close();
    std::cout << "Saved " << nv << " valid tiles to " << filename << std::endl;
}