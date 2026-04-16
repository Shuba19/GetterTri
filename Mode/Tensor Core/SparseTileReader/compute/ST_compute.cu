#include "../../../../Libs/ChronoCuda.h"
#include "../../../../Libs/GraphReader.h"
#include "../../../../Libs/CudaUtilities.h"
#include "../../../../Libs/TensorUtilities.h"
#include "../../../../Libs/TileReader.h"

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


struct output_t
{
    std::string file;
    int triangles = -1,  memory_peak_mb = -1;
    float memory_total_mb = -1;
    float total_time, kernel_time;
    std::string unit_time;
};

void print_output_as_json(const output_t &output)
{
    std::cout << "{\n";
    std::cout << "  \"file\": \"" << output.file << "\",\n";
    std::cout << "  \"triangles\": " << output.triangles << ",\n";
    std::cout << "  \"total_time\": " << output.total_time << ",\n";
    std::cout << "  \"kernel_time\": " << output.kernel_time << ",\n";
    std::cout << "  \"memory_total_mb\": " << output.memory_total_mb << ",\n";
    std::cout << "  \"memory_peak_mb\": " << output.memory_peak_mb << ",\n";
    std::cout << "  \"unit_time\": \"" << output.unit_time << "\"\n";
    std::cout << "}\n";
}


// Cambiato il tipo di res in unsigned long long*
__global__ void square_then_hadamard_warped_sparse(
    int tpr, 
    const tiles_b *__restrict__ matrix, 
    const int64_t *__restrict__ id_tiles, 
    int num_tiles, 
    unsigned long long *__restrict__ res);

output_t TTC_4(TILES &t);

int main(int argc, char *argv[])
{
    chrono_cuda timer("TotalEX");
    timer.cc_start();
    if (argc < 2) return 1;
    TILES t = readTiles(argv[1]);
    output_t output = TTC_4(t);
    timer.cc_stop(false);
    output.file = std::string(argv[1]);
    output.total_time = timer.elapsed;
    print_output_as_json(output);

    return 0;
}

output_t TTC_4(TILES &t)
{
    output_t output;
    cudaSetDevice(0);
    tiles_b *d_matrix;
    int64_t *d_tile_ids;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    int tpr = (t.num_v + 15) >> 4;
    
    CHECK(cudaMallocAsync((void **)&d_matrix, t.tiles.size() * sizeof(tiles_b), stream));
    CHECK(cudaMallocAsync((void **)&d_tile_ids, t.tile_ids.size() * sizeof(int64_t), stream));
    CHECK(cudaMemcpyAsync(d_matrix, t.tiles.data(), t.tiles.size() * sizeof(tiles_b), cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(d_tile_ids, t.tile_ids.data(), t.tile_ids.size() * sizeof(int64_t), cudaMemcpyHostToDevice, stream));

    unsigned long long *d_res;
    // Alloco un solo elemento per il totale (come nel tuo kernel atomicAdd(&res[0], ...))
    // Se invece vuoi un array per tile, cambia la size a t.tiles.size()
    CHECK(cudaMallocAsync((void **)&d_res, sizeof(unsigned long long), stream));
    CHECK(cudaMemsetAsync(d_res, 0, sizeof(unsigned long long), stream));
    cudaStreamSynchronize(stream);
    chrono_cuda timer("TTC_4", stream);
    timer.cc_start();

    dim3 block(32);
    dim3 grid(t.tiles.size(), 1); // Grid cambiata: k_tile è gestito nel loop interno del kernel

    square_then_hadamard_warped_sparse<<<grid, block, 0, stream>>>(tpr, d_matrix, d_tile_ids, (int)t.tiles.size(), d_res);
    int tri;
    CHECK(cudaMemcpyAsync(&tri, d_res, sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamSynchronize(stream));
    timer.cc_stop(false);
    output.kernel_time = timer.elapsed;
    output.triangles = (int)tri / 6;
    float memory_total;
    memory_total = (float)((t.tiles.size() * sizeof(tiles_b) + t.tile_ids.size() * sizeof(int64_t) + sizeof(unsigned long long)) / (1024.0 * 1024.0));
    output.memory_total_mb = memory_total;
    return output;
}

__device__ int bin_search(const int64_t *arr, int size, int64_t target)
{
    int left = 0, right = size - 1;
    while (left <= right)
    {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target) return mid;
        else if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}

__global__ void square_then_hadamard_warped_sparse(
    int tpr, 
    const tiles_b *__restrict__ matrix, 
    const int64_t *__restrict__ id_tiles, 
    int num_tiles, 
    unsigned long long *__restrict__ res)
{
    const int current_tile_idx = blockIdx.x;
    const int64_t tile_id = id_tiles[current_tile_idx];
    const int tid = threadIdx.x;

    __shared__  uint8_t A[256];
    __shared__  uint8_t B[256];
    __shared__  int temp_C[256];

    const int t_col = triangular_col_from_id(tile_id);
    const int t_row = (int)(tile_id - (int64_t)t_col * (t_col + 1) / 2);

    wmma::fragment<wmma::accumulator, 16, 16, 16, int> c_frag;
    wmma::fill_fragment(c_frag, 0);

    for (int k_tile = 0; k_tile < tpr; k_tile++)
    {
        int64_t target1 = (k_tile > t_col) ? from_x_y_to_id(k_tile, t_col) : from_x_y_to_id(t_col, k_tile);
        int64_t target2 = (t_row > k_tile) ? from_x_y_to_id(t_row, k_tile) : from_x_y_to_id(k_tile, t_row);

        int idx1 = bin_search(id_tiles, num_tiles, target1);
        int idx2 = bin_search(id_tiles, num_tiles, target2);

        if (idx1 != -1 && idx2 != -1)
        {
            bool trans1 = (k_tile > t_col);
            bool trans2 = (t_row > k_tile);

            #pragma unroll
            for (int i = tid; i < 256; i += 32)
            {
                const int r = i / 16;
                const int c = i % 16;
                uint16_t val1 = matrix[idx1].tile[trans1 ? c : r];
                A[i] = (val1 >> (15 - (trans1 ? r : c))) & 1u;

                uint16_t val2 = matrix[idx2].tile[trans2 ? c : r];
                B[i] = (val2 >> (15 - (trans2 ? r : c))) & 1u;
            }

            __syncwarp();
            wmma::fragment<wmma::matrix_a, 16, 16, 16, unsigned char, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, unsigned char, wmma::row_major> b_frag;
            
            wmma::load_matrix_sync(a_frag, A, 16);
            wmma::load_matrix_sync(b_frag, B, 16);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            __syncwarp();
        }
    }

    wmma::store_matrix_sync(temp_C, c_frag, 16, wmma::mem_row_major);
    __syncwarp();

    unsigned long long s = 0;
    for (int i = tid; i < 256; i += 32)
    {
        const int r = i / 16;
        const int c = i % 16;
        if ((matrix[current_tile_idx].tile[r] >> (15 - c)) & 1) {
            s += (unsigned int)temp_C[i];
        }
    }

    for (int offset = 16; offset > 0; offset /= 2)
        s += __shfl_down_sync(0xffffffff, s, offset);

    if (tid == 0)
    {
        int factor = (t_row != t_col) ? 2 : 1;
        atomicAdd(res, s * (unsigned long long)factor);
    }
}