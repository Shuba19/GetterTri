#ifndef GRAPHSTATS
#define GRAPHSTATS
#include "CudaUtilities.h"



struct GraphStats
{
    int deg_max = 0;
    int log_maj_deg_max = 0;
    int deg_maj_threshold = 0;
    bool need_help =false;
};

void reorder_graph( GraphData &graph, graph_device device);
GraphStats calculate_stats_graph(const graph_device &device);


int linear_reg_threshold(const GraphStats &stats);


#endif