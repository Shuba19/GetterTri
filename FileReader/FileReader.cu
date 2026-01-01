#include "FileReader.h"

/*********************************
 * Metis Graph first line parameters:
 *
 * - V    : number of vertexes
 * - E    : number of edges
 * - fmt  : can have four values (0 no weight, 1: edges weighted, 10: vertices weighted, 11: edges and vertices weighted)
 * - ncon : number of weight associated to vertices
 /*******************************/


#define REP_BENCHMARK 100
GraphFR::~GraphFR()
{
}

GraphFR::GraphFR(const CommandArgs &args) : args(args)
{
  cudaEventCreate(&this->timer.t1);
  cudaEventCreate(&this->timer.t2);
}

bool GraphFR::IsMetisComment(const std::string &str)
{
  for (char c : str)
  {
    if (c == '%')
      return true;
  }
  return false;
}

bool GraphFR::GraphReader(std::ifstream &GraphInput, bool e_weight, bool v_weight, int n_skip)
{
  std::string line;
  int pos = 0;
  offsets.push_back(0);
  int sum = 0;
  while (std::getline(GraphInput, line) && pos < num_v)
  {
    if (IsMetisComment(line))
      continue;
    std::istringstream iss(line);
    int garbage;
    if (v_weight)
    {
      iss >> garbage;
      for (int i = 1; i < n_skip; i++)
        iss >> garbage;
    }
    int edge;
    std::vector<int> v_edge;
    while (iss >> edge)
    {
      v_edge.push_back(edge - 1);
      if (e_weight)
        iss >> garbage;
    }
    std::sort(v_edge.begin(), v_edge.end());
    sum = sum + v_edge.size();
    offsets.push_back(sum);
    if (!v_edge.empty())
    {
      for (auto i : v_edge)
        csr.push_back(i);
    }
    pos++;
  }
  return true;
}

bool GraphFR::ReadFile()
{
  std::ifstream GraphInput(this->args.input_file);
  if (!GraphInput.is_open())
  {
    std::cerr << "Error Opening File, check if the file exists or/and if the name is correct." << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string line;
  do
  {
    std::getline(GraphInput, line);
  } while (IsMetisComment(line) && !GraphInput.eof());
  if (GraphInput.eof())
    return false;

  std::vector<int> args(4);
  args[0] = 0;
  args[1] = 0;
  args[2] = 0;
  args[3] = 0;
  std::istringstream iss(line);
  int value;
  int p = 0;
  while (iss >> value)
  {
    args[p++] = value;
  }
  bool e_w = (args[2] & 1);
  bool v_w = (args[2] & 2);
  this->num_v = args[0];
  this->num_edge = args[1];
  bool result = GraphReader(GraphInput, e_w, v_w, args[3]);
  GraphInput.close();
  return result;
}

void GraphFR::StartTimer()
{
  cudaEventRecord(this->timer.t1, 0);
}
void GraphFR::StopTimer()
{
  cudaEventRecord(this->timer.t2, 0);
  cudaEventSynchronize(this->timer.t2);
  cudaEventElapsedTime(&this->timer.time, this->timer.t1, this->timer.t2);
}

void GraphFR::printVerboseGraphInfo()
{
  if (!this->args.verbose)
    return;
  double num_possible_edges = static_cast<double>(this->num_v) * (this->num_v - 1);
  double density = static_cast<double>(this->num_edge) / num_possible_edges;

  if (this->args.undirect)
  {
    density *= 2.0;
  }

  std::cout << "----------------------------------" << std::endl;
  std::cout << "Graph Information:" << std::endl;
  std::cout << "Number of vertices: " << this->num_v << std::endl;
  std::cout << "Number of edges: " << this->num_edge << std::endl;
  std::cout << "Graph Density: " << (density * 100.0) << "%" << std::endl;
  std::cout << "CSR Size: " << this->csr.size() << std::endl;
  std::cout << "Offsets Size: " << this->offsets.size() << std::endl;
  std::cout << "Mode: " << (TriMode)this->args.mode << std::endl;
  if (this->args.benchmark)
  {
    std::cout << "Benchmarking mode enabled with " << REP_BENCHMARK << " repetitions." << std::endl;
    std::cout << "Average time per operation: " << this->timer.time / REP_BENCHMARK << " ms" << std::endl;
  }
  else
    std::cout << "Time taken for last operation: " << this->timer.time << " ms" << std::endl;
  std::cout << "----------------------------------" << std::endl; 
}

out_type GraphFR::CalculateTriangles()
{
  if (this->args.benchmark)
  {
    benchmark();
    return 0;
  }
  out_type triangle_count = 0;
  StartTimer();
  switch (this->args.mode)
  {
  case 0:
    triangle_count = SearchTriangle_Edge_Iterator(this->num_v, this->num_edge, this->offsets, this->csr, this->args.undirect);
    break;
  case 1:
    triangle_count = SearchTriangle_Node_Iterator(this->num_v, this->num_edge, this->offsets, this->csr, this->args.undirect);
    break;
  case 2:
    triangle_count = TTC(this->num_v, this->num_edge, this->offsets, this->csr);
    break;
  case 3:
    triangle_count = triangle_couting_CPU(this->num_v, this->num_edge, this->offsets, this->csr);
    break;  
  default:
    std::cerr << "Invalid mode selected. Please choose 'n' for Node Iterator, 'e' for Edge Iterator, or 't' for Tensor Calculation." << std::endl;
    return -1;
  }
  StopTimer();
  printVerboseGraphInfo();
  return triangle_count;
}

void GraphFR::benchmark()
{
  std::cout << "------- STARTING BENCHMARK -------" << std::endl;
  SearchTriangle_Edge_Iterator(this->num_v, this->num_edge, this->offsets, this->csr, this->args.undirect);
  StartTimer();

  switch (this->args.mode)
  {
  case 0:
    for (int i = 0; i < REP_BENCHMARK; i++)
      SearchTriangle_Edge_Iterator(this->num_v, this->num_edge, this->offsets, this->csr, this->args.undirect);
    break;
  case 1:
    for (int i = 0; i < REP_BENCHMARK; i++)
      SearchTriangle_Node_Iterator(this->num_v, this->num_edge, this->offsets, this->csr, this->args.undirect);
    break;
  case 2:
    for (int i = 0; i < REP_BENCHMARK; i++)
      TTC(this->num_v, this->num_edge, this->offsets, this->csr);
  case 3:
    for (int i = 0; i < REP_BENCHMARK; i++)
      triangle_couting_CPU(this->num_v, this->num_edge, this->offsets, this->csr);
    break;
    break;
  default:
    std::cerr << "Invalid mode selected. Please choose 'n' for Node Iterator, 'e' for Edge Iterator, or 't' for Tensor Calculation." << std::endl;
    return;
  }
  StopTimer();
  printVerboseGraphInfo();
  std::cout << "------- END OF BENCHMARK ---------" << std::endl;
}
