#include "../CommonMethods/common_methods.h"

std::mutex mtx;
out_type count_tri = 0;

void add_tri(int n_tri)
{
    {
        std::lock_guard<std::mutex> lock(mtx);
        count_tri += n_tri;
    }
}

out_type triangle_couting_CPU(int num_v, int64_t n_edges, const std::vector<int> &offsets, const std::vector<int> &csr)
{
    count_tri = 0;
#pragma omp parallel for
    for (int edge = 0; edge < csr.size(); edge++)
    {
        int n_tri = 0;
        auto it = std::upper_bound(offsets.begin(), offsets.end(), edge);
        int s_node = std::distance(offsets.begin(), it);
        int d_node = csr[edge];
        if (s_node >= d_node)
            continue;

        int s1 = offsets[s_node], e1 = offsets[s_node + 1];
        int s2 = offsets[d_node], e2 = offsets[d_node + 1];
        while (s1 < e1 && s2 < e2)
        {
            int c1 = csr[s1], c2 = csr[s2];
            if (c1 == c2)
            {
                if (c1 > d_node)
                {
                    n_tri++;
                }
                s1++;
                s2++;
            }
            else if (c1 < c2)
                s1++;
            else
                s2++;
        }
        add_tri(n_tri);
    }
    return count_tri;
}
