#ifndef GRAPHFILEREADER
#define GRAPHFILEREADER
#include "command_args.h"
#include <string>
#include <iostream>
#include <fstream>
#include <sys/mman.h>
#include <sstream>
#include <vector>
#include <algorithm>
#include "Libs/CommonMethods/common_methods.h"

enum TriMode
{
  EDGE_ITERATOR = 0,
  NODE_ITERATOR = 1,
  TENSOR_CALCULATION = 2,
  OPENMP = 3
};
struct timerEvent{
    cudaEvent_t t1,t2;
    float time;
};

class GraphFR{
    int num_v, num_edge;
    std::vector<int> csr, offsets;
    int numArgs;
    CommandArgs args;
    timerEvent timer;
    void StartTimer();
    void StopTimer();
    bool GraphReader(std::ifstream& GraphInput, bool e_weight, bool v_weight, int n_skip);
    bool IsMetisComment(const std::string& str);
    void printVerboseGraphInfo();
    void benchmark();
    public:
    GraphFR(const CommandArgs& args);
    ~GraphFR();
    bool ReadFile();
    out_type CalculateTriangles();
};

#endif