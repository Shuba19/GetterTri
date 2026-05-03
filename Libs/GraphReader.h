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
#include <vector>
#include <string>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <thread>
#include <atomic>
#include <numeric>

struct GraphData
{
    int num_v, num_edge;
    std::vector<int> csr, s_edge, offsets;
    std::vector<std::pair<int,int>> deg;
};

struct output_t
{
    std::string file;
    int64_t triangles,num_v,num_e;
    double density;
    int memory_total, memory_peak;
    float total_time, kernel_time, read_time, preprocess_time;
    std::string unit_time, unit_memory;
    std::map<int, std::map<int, float>> time_per_threshold;
};

void print_output_as_json(const output_t &output);

GraphData readGraph(const std::string &filename);
GraphData readGraph_Forward(const std::string &filename);
GraphData readGraph_Tile(const std::string &filename);

#endif