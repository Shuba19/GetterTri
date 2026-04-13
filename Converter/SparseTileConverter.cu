#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>

#define TILE_SIDE             16
#define TILE_BUILDER_THREADS  128
#define TILE_ROWS_PER_GROUP   16
#define TILE_GROUPS_PER_BLOCK (TILE_BUILDER_THREADS / TILE_ROWS_PER_GROUP)
#define BATCH_TILES           (102400000LL)
#define MAX_LINE_BYTES        128  // 19 + 16*6 + 1 < 128

using namespace std;

struct tiles_b
{
    uint16_t tile[TILE_SIDE];
};

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

// O(1) invece di O(sqrt(n)) loop
__device__ int triangular_col_from_id(long long id)
{
    int col = (int)((sqrtf(8.0f * (float)id + 1.0f) - 1.0f) * 0.5f);
    while ((long long)col * (col + 1) / 2 > id)        --col;
    while ((long long)(col+1) * (col+2) / 2 <= id)     ++col;
    return col;
}

__device__ __forceinline__ int lower_bound_device(const int *__restrict__ values, int begin, int end, int key)
{
    int left = begin, right = end;
    while (left < right)
    {
        int mid = left + ((right - left) >> 1);
        if (values[mid] < key) left  = mid + 1;
        else                   right = mid;
    }
    return left;
}

__device__ int dev_write_uint(char *buf, unsigned int val)
{
    if (val == 0) { buf[0] = '0'; return 1; }
    char tmp[6];
    int len = 0;
    while (val > 0) { tmp[len++] = '0' + (val % 10); val /= 10; }
    for (int i = 0; i < len; i++) buf[i] = tmp[len - 1 - i];
    return len;
}

__device__ int dev_write_ll(char *buf, long long val)
{
    if (val == 0) { buf[0] = '0'; return 1; }
    char tmp[20];
    int len = 0;
    while (val > 0) { tmp[len++] = '0' + (val % 10); val /= 10; }
    for (int i = 0; i < len; i++) buf[i] = tmp[len - 1 - i];
    return len;
}

// ---------------------------------------------------------------------------
// Kernel
//
// Ogni thread calcola il suo row_bits e lo scrive direttamente in d_out.
// Thread 0 di ogni slot controlla se il tile è non-empty e serializza
// la riga in d_strbuf (buffer di stringhe pronto per fwrite).
// d_non_empty[local_tile_id] = 1 se il tile ha almeno un bit settato.
// ---------------------------------------------------------------------------
__global__ void tiles_builder(
    int              num_v,
    long long        batch_start,
    int              batch_size,
    const int       *__restrict__ csr,
    const int       *__restrict__ ofs,
    tiles_b         *__restrict__ d_out,
    char            *__restrict__ d_strbuf,   // [batch_size * MAX_LINE_BYTES]
    int             *__restrict__ d_linelen)  // [batch_size] lunghezza stringa per tile
{
    const int global_thread = blockDim.x * blockIdx.x + threadIdx.x;
    const int local_tile_id = global_thread / TILE_ROWS_PER_GROUP;
    const int local_row     = threadIdx.x & (TILE_ROWS_PER_GROUP - 1);
    const int tile_slot     = threadIdx.x / TILE_ROWS_PER_GROUP;

    // Shared: una riga per slot per la riduzione non-empty
    __shared__ uint16_t s_tile[TILE_GROUPS_PER_BLOCK][TILE_SIDE];
    __shared__ int      shared_tile_col[TILE_GROUPS_PER_BLOCK];
    __shared__ int      shared_tile_row[TILE_GROUPS_PER_BLOCK];

    // Inizializza shared a 0
    s_tile[tile_slot][local_row] = 0;

    if (local_row == 0 && local_tile_id < batch_size)
    {
        const long long tile_id = batch_start + local_tile_id;
        const int tile_col = triangular_col_from_id(tile_id);
        shared_tile_col[tile_slot] = tile_col;
        shared_tile_row[tile_slot] = (int)(tile_id - (long long)tile_col * (tile_col + 1) / 2);
    }
    __syncthreads();

    if (local_tile_id >= batch_size)
        return;

    const int tile_col = shared_tile_col[tile_slot];
    const int tile_row = shared_tile_row[tile_slot];
    const int start_x  = tile_col * TILE_SIDE;
    const int y        = tile_row * TILE_SIDE + local_row;

    uint16_t row_bits = 0;
    if (y < num_v && start_x < num_v)
    {
        const int row_begin = ofs[y];
        const int row_end   = ofs[y + 1];
        const int col_limit = min(start_x + TILE_SIDE, num_v);
        int pos = lower_bound_device(csr, row_begin, row_end, start_x);
        while (pos < row_end)
        {
            const int x = csr[pos++];
            if (x >= col_limit) break;
            row_bits |= (uint16_t)(1u << (TILE_SIDE - 1 - (x - start_x)));
        }
    }

    // Ogni thread scrive il suo row direttamente in global memory (coalescente)
    d_out[local_tile_id].tile[local_row] = row_bits;
    // e in shared per il controllo non-empty
    s_tile[tile_slot][local_row] = row_bits;
    __syncthreads();

    // Thread 0 dello slot: check non-empty + serializza stringa
    if (local_row == 0)
    {
        bool non_empty = false;
        for (int i = 0; i < TILE_SIDE; ++i)
            if (s_tile[tile_slot][i]) { non_empty = true; break; }

        if (non_empty)
        {
            const long long tile_id = batch_start + local_tile_id;
            char *line = d_strbuf + (long long)local_tile_id * MAX_LINE_BYTES;
            int len = 0;
            len += dev_write_ll(line + len, tile_id);
            for (int i = 0; i < TILE_SIDE; ++i)
            {
                line[len++] = ' ';
                len += dev_write_uint(line + len, s_tile[tile_slot][i]);
            }
            line[len++] = '\n';
            d_linelen[local_tile_id] = len;
        }
        else
        {
            d_linelen[local_tile_id] = 0;  // empty tile, skip
        }
    }
}

// ---------------------------------------------------------------------------
// METIS reader
// ---------------------------------------------------------------------------
bool read_metis_graph(const string &path, int &num_vertices, int &num_edges,
                      vector<int> &csr, vector<int> &off)
{
    num_vertices = num_edges = 0;
    ifstream graph(path);
    if (!graph.is_open()) return false;

    string line;
    bool header_read = false;
    while (getline(graph, line))
    {
        if (line.empty() || line[0] == '%') continue;
        istringstream ss(line);
        ss >> num_vertices >> num_edges;
        header_read = !ss.fail();
        break;
    }
    if (!header_read || num_vertices <= 0) return false;

    off.assign((size_t)num_vertices + 1, 0);
    csr.reserve((size_t)num_edges * 2);

    int row = 0;
    while (row < num_vertices && getline(graph, line))
    {
        if (line.empty() || line[0] == '%') continue;
        vector<int> neighbors;
        istringstream ss(line);
        int edge;
        while (ss >> edge)
        {
            const int zb = edge - 1;
            if (zb < 0 || zb >= num_vertices || zb == row) continue;
            neighbors.push_back(zb);
        }
        sort(neighbors.begin(), neighbors.end());
        neighbors.erase(unique(neighbors.begin(), neighbors.end()), neighbors.end());
        csr.insert(csr.end(), neighbors.begin(), neighbors.end());
        off[row + 1] = (int)csr.size();
        ++row;
    }
    while (row < num_vertices) { off[row + 1] = (int)csr.size(); ++row; }
    return true;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        cerr << "Usage: " << argv[0] << " <graph.metis> <output.stile>\n";
        return 1;
    }

    vector<int> h_csr, h_off;
    int v = 0, e = 0;
    if (!read_metis_graph(argv[1], v, e, h_csr, h_off))
    {
        cerr << "Cannot read METIS graph from: " << argv[1] << '\n';
        return 1;
    }

    const long long tpr      = (v + TILE_SIDE - 1) / TILE_SIDE;
    const long long total_t  = tpr * (tpr + 1) / 2;
    const long long num_iter = (total_t + BATCH_TILES - 1) / BATCH_TILES;

    cerr << "vertices: " << v << "  edges: " << e
         << "  tpr: " << tpr << "  total_tiles: " << total_t
         << "  batches: " << num_iter << '\n';

    // Upload CSR una volta sola
    int *d_csr, *d_off;
    cudaMalloc(&d_csr, h_csr.size() * sizeof(int));
    cudaMalloc(&d_off, h_off.size() * sizeof(int));
    cudaMemcpy(d_csr, h_csr.data(), h_csr.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_off, h_off.data(), h_off.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Buffer device (allocati una volta, riusati per ogni batch)
    tiles_b *d_out     = nullptr;
    char    *d_strbuf  = nullptr;
    int     *d_linelen = nullptr;
    cudaMalloc(&d_out,     BATCH_TILES * sizeof(tiles_b));
    cudaMalloc(&d_strbuf,  BATCH_TILES * MAX_LINE_BYTES);
    cudaMalloc(&d_linelen, BATCH_TILES * sizeof(int));

    // Pinned memory host per overlap memcpy/scrittura
    tiles_b *h_out    = nullptr;
    char    *h_strbuf = nullptr;
    int     *h_linelen = nullptr;
    cudaMallocHost(&h_out,     BATCH_TILES * sizeof(tiles_b));
    cudaMallocHost(&h_strbuf,  BATCH_TILES * MAX_LINE_BYTES);
    cudaMallocHost(&h_linelen, BATCH_TILES * sizeof(int));

    // Due stream: uno per il kernel, uno per il memcpy asincrono
    cudaStream_t stream_kernel, stream_copy;
    cudaStreamCreate(&stream_kernel);
    cudaStreamCreate(&stream_copy);

    // Buffer di scrittura su file (un unico fwrite per batch)
    vector<char> file_buf;
    file_buf.reserve((size_t)BATCH_TILES * MAX_LINE_BYTES / 4); // stima conservativa

    FILE *fout = fopen(argv[2], "w");
    if (!fout) { cerr << "Cannot open output: " << argv[2] << '\n'; return 1; }

    // Riserva 64 byte fissi per l'header in cima al file.
    // Formato finale: "num_v num_e num_valid_tiles\n" con padding di spazi.
    // Alla fine faremo fseek(0) e sovrascriveremo.
    const int HEADER_SIZE = 64;
    char header_placeholder[64];
    memset(header_placeholder, ' ', HEADER_SIZE);
    header_placeholder[HEADER_SIZE - 1] = '\n';
    fwrite(header_placeholder, 1, HEADER_SIZE, fout);

    long long total_valid = 0;

    for (long long iter = 0; iter < num_iter; ++iter)
    {
        const long long batch_start = iter * BATCH_TILES;
        const int       batch_size  = (int)min(BATCH_TILES, total_t - batch_start);

        // Pulisci linelen per questo batch
        cudaMemsetAsync(d_linelen, 0, batch_size * sizeof(int), stream_kernel);

        const int num_blocks = (batch_size * TILE_ROWS_PER_GROUP + TILE_BUILDER_THREADS - 1)
                               / TILE_BUILDER_THREADS;

        tiles_builder<<<num_blocks, TILE_BUILDER_THREADS, 0, stream_kernel>>>(
            v, batch_start, batch_size,
            d_csr, d_off,
            d_out, d_strbuf, d_linelen);

        // Copia strbuf e linelen sul host (asincrono sullo stesso stream)
        cudaMemcpyAsync(h_strbuf,  d_strbuf,  (size_t)batch_size * MAX_LINE_BYTES, cudaMemcpyDeviceToHost, stream_kernel);
        cudaMemcpyAsync(h_linelen, d_linelen,  (size_t)batch_size * sizeof(int),   cudaMemcpyDeviceToHost, stream_kernel);

        cudaStreamSynchronize(stream_kernel);

        // Costruisci file_buf: concatena solo le righe non-empty
        file_buf.clear();
        for (int i = 0; i < batch_size; ++i)
        {
            const int len = h_linelen[i];
            if (len > 0)
            {
                const char *line = h_strbuf + (long long)i * MAX_LINE_BYTES;
                file_buf.insert(file_buf.end(), line, line + len);
                ++total_valid;
            }
        }

        // Un solo fwrite per tutto il batch
        if (!file_buf.empty())
            fwrite(file_buf.data(), 1, file_buf.size(), fout);

        cerr << "batch " << (iter + 1) << "/" << num_iter
             << "  valid so far: " << total_valid << "\r";
    }
    cerr << '\n';

    // Torna all'inizio e sovrascrivi l'header con i valori reali.
    // Formato: "num_v num_e num_valid_tiles" paddato con spazi fino a 63 chars + '\n'
    fseek(fout, 0, SEEK_SET);
    char header[64];
    int hlen = snprintf(header, sizeof(header), "%d %d %lld", v, e, total_valid);
    // Pad con spazi fino a 63, poi newline
    memset(header + hlen, ' ', 63 - hlen);
    header[63] = '\n';
    fwrite(header, 1, 64, fout);

    fclose(fout);
    cerr << "total valid tiles: " << total_valid << '\n';

    cudaFree(d_csr);
    cudaFree(d_off);
    cudaFree(d_out);
    cudaFree(d_strbuf);
    cudaFree(d_linelen);
    cudaFreeHost(h_out);
    cudaFreeHost(h_strbuf);
    cudaFreeHost(h_linelen);
    cudaStreamDestroy(stream_kernel);
    cudaStreamDestroy(stream_copy);
    return 0;
}
