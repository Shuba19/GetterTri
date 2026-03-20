#include "edge_iterator_solver.h"
void filter_per_deg(const std::vector<int> &ofs, const std::vector<int> &s_edges, std::vector<int> &thread_level, std::vector<int> &warp_level)
{
    int nthreads = omp_get_max_threads();
    std::vector<std::vector<int>> local_thread(nthreads);
    std::vector<std::vector<int>> local_warp(nthreads);

#pragma omp parallel
    {
        int tid = omp_get_thread_num();

#pragma omp for
        for (size_t i = 0; i < s_edges.size(); ++i)
        {
            int s = s_edges[i];
            int d = ofs[s + 1] - ofs[s];

            if (d <= THRESHOLD)
                local_thread[tid].push_back(i);
            else
                local_warp[tid].push_back(i);
        }
    }

    for (int t = 0; t < nthreads; ++t)
    {
        thread_level.insert(thread_level.end(),
                            local_thread[t].begin(),
                            local_thread[t].end());

        warp_level.insert(warp_level.end(),
                          local_warp[t].begin(),
                          local_warp[t].end());
    }
}

__device__ __forceinline__ bool bin_search(const int *csr, int s, int e, int key)
{
    int l = s, r = e - 1;
    while (l <= r)
    {
        int m = (l + r) >> 1;
        int v = csr[m];
        if (v == key)
            return true;
        if (v < key)
            l = m + 1;
        else
            r = m - 1;
    }
    return false;
}

__device__ __forceinline__ int upper_bound(const int *csr, int s, int e, int key)
{
    int l = s, r = e;
    while (l < r)
    {
        int m = (l + r) >> 1;
        if (csr[m] <= key)
            l = m + 1;
        else
            r = m;
    }
    return l;
}
// Default Kernel for edge iterator
__global__ void edge_search_tri(int num_v, int64_t num_e, const int *__restrict__ ofs, const int *__restrict__ csr, const int *__restrict__ s_edge, int *__restrict__ results)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_e)
        return;

    int s_node = s_edge[id];
    int d_node = csr[id];
    if (d_node <= s_node)
    {
        results[id] = 0;
        return;
    }

    int s1 = ofs[s_node], e1 = ofs[s_node + 1];
    int s2 = ofs[d_node], e2 = ofs[d_node + 1];

    s1 = upper_bound(csr, s1, e1, d_node);
    s2 = upper_bound(csr, s2, e2, d_node);

    int len1 = e1 - s1;
    int len2 = e2 - s2;

    if (len1 == 0 || len2 == 0)
    {
        results[id] = 0;
        return;
    }

    int n_tri = 0;

    int ss, se, ls, le;
    if (len1 <= len2)
    {
        ss = s1;
        se = e1;
        ls = s2;
        le = e2;
    }
    else
    {
        ss = s2;
        se = e2;
        ls = s1;
        le = e1;
    }
    int short_len = se - ss;
    int long_len = le - ls;

    if (long_len <= short_len * 16)
    {
        int i = ss, j = ls;
        while (i < se && j < le)
        {
            int a = csr[i], b = csr[j];
            n_tri += (a == b);
            i += (a <= b);
            j += (a >= b);
        }
    }
    else
    {
        for (int i = ss; i < se; ++i)
        {
            n_tri += bin_search(csr, ls, le, csr[i]);
        }
    }

    results[id] = n_tri;
}

// thread level kernel
__global__ void edge_thread_search_tri(int num_v, int64_t num_e, const int *__restrict__ ofs, const int *__restrict__ csr, const int *__restrict__ s_edge, int *__restrict__ results, const int *thread_level)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_e)
        return;

    int s_node = s_edge[thread_level[id]];
    int d_node = csr[thread_level[id]];
    if (d_node <= s_node)
    {
        results[thread_level[id]] = 0;
        return;
    }

    int s1 = ofs[s_node], e1 = ofs[s_node + 1];
    int s2 = ofs[d_node], e2 = ofs[d_node + 1];

    s1 = upper_bound(csr, s1, e1, d_node);
    s2 = upper_bound(csr, s2, e2, d_node);

    int len1 = e1 - s1;
    int len2 = e2 - s2;

    if (len1 == 0 || len2 == 0)
    {
        results[thread_level[id]] = 0;
        return;
    }

    int n_tri = 0;

    int ss, se, ls, le;
    if (len1 <= len2)
    {
        ss = s1;
        se = e1;
        ls = s2;
        le = e2;
    }
    else
    {
        ss = s2;
        se = e2;
        ls = s1;
        le = e1;
    }
    int short_len = se - ss;
    int long_len = le - ls;

    if (long_len <= short_len * 16)
    {
        int i = ss, j = ls;
        while (i < se && j < le)
        {
            int a = csr[i], b = csr[j];
            n_tri += (a == b);
            i += (a <= b);
            j += (a >= b);
        }

        results[thread_level[id]] = n_tri;
    }
    else
    {
        for (int i = ss; i < se; ++i)
        {
            n_tri += bin_search(csr, ls, le, csr[i]);
        }
        results[thread_level[id]] = n_tri;
    }
}

// warp level kernel
__global__ void edge_warp_search_tri(int num_v, int64_t num_e, const int *__restrict__ ofs, const int *__restrict__ csr, const int *__restrict__ s_edge, int *__restrict__ results, const int *warp_level)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = id / WARP_SIZE;
    int lane = id % WARP_SIZE;

    if (warp_id >= num_e)
        return;

    int edge_idx = warp_level[warp_id];
    int s_node = s_edge[edge_idx];
    int d_node = csr[edge_idx];

    if (d_node <= s_node)
    {
        if (lane == 0)
            results[edge_idx] = 0;
        return;
    }

    int s1 = ofs[s_node], e1 = ofs[s_node + 1];
    int s2 = ofs[d_node], e2 = ofs[d_node + 1];

    s1 = upper_bound(csr, s1, e1, d_node);
    s2 = upper_bound(csr, s2, e2, d_node);

    int len1 = e1 - s1;
    int len2 = e2 - s2;

    if (len1 == 0 || len2 == 0)
    {
        if (lane == 0)
            results[edge_idx] = 0;
        return;
    }

    int ss, se, ls, le;
    if (len1 <= len2)
    {
        ss = s1;
        se = e1;
        ls = s2;
        le = e2;
    }
    else
    {
        ss = s2;
        se = e2;
        ls = s1;
        le = e1;
    }

    int n_tri = 0;
    for (int i = ss + lane; i < se; i += WARP_SIZE)
    {
        n_tri += bin_search(csr, ls, le, csr[i]);
    }
    unsigned mask = 0xffffffff;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    {
        n_tri += __shfl_down_sync(mask, n_tri, offset);
    }

    if (lane == 0)
        results[edge_idx] = n_tri;
}

// basically a edge_iterator, but once finished helps the remaining threads to finish their work, to reduce the load imbalance
// WIP WIP WIP WIP IWP
__global__ void help_search_tri(int num_v, int64_t num_e, const int *__restrict__ ofs, const int *__restrict__ csr, const int *__restrict__ s_edge, int *__restrict__ results)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_e)
        return;
    int s_node = s_edge[id];
    int d_node = csr[id];
    if (d_node <= s_node)
    {
        results[id] = 0;
        return;
    }

    int s1 = ofs[s_node], e1 = ofs[s_node + 1];
    int s2 = ofs[d_node], e2 = ofs[d_node + 1];

    s1 = upper_bound(csr, s1, e1, d_node);
    s2 = upper_bound(csr, s2, e2, d_node);

    int len1 = e1 - s1;
    int len2 = e2 - s2;

    if (len1 == 0 || len2 == 0)
    {
        results[id] = 0;
        return;
    }

    int n_tri = 0;

    int ss, se, ls, le;
    if (len1 <= len2)
    {
        ss = s1;
        se = e1;
        ls = s2;
        le = e2;
    }
    else
    {
        ss = s2;
        se = e2;
        ls = s1;
        le = e1;
    }

    for (int i = ss; i < se; ++i)
    {
        n_tri += bin_search(csr, ls, le, csr[i]);
    }
    results[id] = n_tri;
}