#include "../CommonMethods/common_methods.h"
using namespace nvcuda;

// versione che usa 32 thread per blocco

#define CHECK(call)                                                   \
  {                                                                   \
    const cudaError_t error = call;                                   \
    if (error != cudaSuccess)                                         \
    {                                                                 \
      printf("Error %s : %d\n", __FILE__, __LINE__);                  \
      printf("code:%d, reason:%s", error, cudaGetErrorString(error)); \
      exit(1);                                                        \
    }                                                                 \
  }


  __device__ int tile_bin_search(int goal, int *v_tiles, int len) {
    int left = 0;
    int right = len - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (v_tiles[mid] == goal) {
            return mid; 
        } else if (v_tiles[mid] < goal) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return 0; 
  }


__global__ void ttc_4_kernel(const tiles_b *__restrict__ matrix, int *__restrict__ v_tiles, int *__restrict__ res, int len, int total_tiles)
{
  int tile_id = blockIdx.x;
  int k_tile = blockIdx.y;

  const int t_col = triangular_col_from_id(tile_id);
  const int t_row = tile_id - t_col * (t_col + 1) / 2;

  int id1, id2;
  int s = 0;
  id1 = (k_tile > t_col) ? from_x_y_to_id(k_tile, t_col) : from_x_y_to_id(t_col, k_tile);
  id2 = (t_row > k_tile) ? from_x_y_to_id(t_row, k_tile) : from_x_y_to_id(k_tile, t_row);

  int t1, t2;
  t1 = (t_col > k_tile);
  t2 = (k_tile > t_row);

  __shared__ bool valid;

  // cerco se esistono i tile id1 e id2, se non esistono esco
  if (threadIdx.x == 0)
  {
    valid = false;
    if (id1 >= 0 && id1 < total_tiles && id2 >= 0 && id2 < total_tiles)
    {
      int bv1 = tile_bin_search(id1, v_tiles, len);
      int bv2 = tile_bin_search(id2, v_tiles, len);
      valid = (bv1 && bv2);
    }
  }
  __syncwarp();
  if(!valid) 
    printf("Invalid tile pair: (%d, %d) with k_tile %d\n", id1, id2, k_tile);
  if (valid)
  {
    // carico i tile id1 e id2 in memoria condivisa
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
#pragma unroll
    for (int i = 0; i < 256; i += 32)
    {
      const int idx = i + threadIdx.x;
      const int r = idx / 16;
      const int c = idx % 16;
      s += (unsigned long long)temp_C[idx] * ((matrix[tile_id].tile[c] >> (15 - r)) & 1);
    }
    s *= factor;
  }
  atomicAdd(&res[tile_id], s);
}

out_type TTC_4(int num_v, int64_t n_edges, const std::vector<tiles_b> &tiles, const std::vector<int> &v_tiles)
{
  int tpr = (num_v + 15) / 16;
  int64_t total_tiles = tiles.size();

  int *d_v_tiles, *d_res;
  tiles_b *d_tiles;
  unsigned long long *d_out;
  out_type h_out = 0;

  CHECK(cudaMalloc(&d_v_tiles, v_tiles.size() * sizeof(int)));
  CHECK(cudaMalloc(&d_res, total_tiles * sizeof(int)));
  CHECK(cudaMalloc(&d_tiles, tiles.size() * sizeof(tiles_b)));
  CHECK(cudaMalloc(&d_out, sizeof(unsigned long long)));

  CHECK(cudaMemcpy(d_v_tiles, v_tiles.data(), v_tiles.size() * sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_tiles, tiles.data(), tiles.size() * sizeof(tiles_b), cudaMemcpyHostToDevice));
  CHECK(cudaMemset(d_res, 0, total_tiles * sizeof(int)));
  CHECK(cudaMemset(d_out, 0, sizeof(unsigned long long)));

  dim3 block(32);
  dim3 grid(total_tiles, tpr);
  // debug
  std::cout << "Launching kernel with grid (" << grid.x << ", " << grid.y << ") and block (" << block.x << ", " << block.y << ", " << block.z << ")" << std::endl;
  std::cout << "Total tiles: " << total_tiles << ", Tiles per row: " << tpr << std::endl;
  std::cout << "Number of vertices: " << num_v << ", Number of edges: " << n_edges << std::endl;
  std::cout << "Size of v_tiles: " << v_tiles.size() << std::endl;
  ttc_4_kernel<<<grid, block>>>(d_tiles, d_v_tiles, d_res, v_tiles.size(), total_tiles);
  CHECK(cudaGetLastError());

  dim3 block2(128);
  dim3 grid2((total_tiles + block2.x - 1) / block2.x);
   reduce_vector<<<grid2, block2>>>(total_tiles, d_res, d_out);
  CHECK(cudaGetLastError());

   CHECK(cudaMemcpy(&h_out, d_out, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

  // free memory
  CHECK(cudaFree(d_v_tiles));
  CHECK(cudaFree(d_res));
  CHECK(cudaFree(d_tiles));
  CHECK(cudaFree(d_out));

  return h_out;
}