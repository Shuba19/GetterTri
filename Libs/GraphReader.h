#ifndef GRAPHREADER
#define GRAPHREADER
#include <string>
#include <iostream>
#include <fstream>
#include <sys/mman.h>
#include <sstream>
#include <vector>
#include <algorithm>
#include <map>
#include <set>


struct GraphData{
    int num_v, num_edge;
    std::vector<int> csr,s_edge, offsets;
};

GraphData readGraph(const std::string& filename);
GraphData readGraph_Forward(const std::string& filename);
GraphData readGraph_Tile(const std::string& filename);

#endif