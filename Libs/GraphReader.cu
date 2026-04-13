#include "GraphReader.h"
bool IsMetisComment(const std::string &str)
{
  for (char c : str)
  {
    if (c == '%')
      return true;
  }
  return false;
}

GraphData readGraph(const std::string &filename)
{
  GraphData graph_data = {0, 0, {}, {}, {}};

  std::ifstream GraphInput(filename);
  if (!GraphInput.is_open())
  {
    std::cerr << "Error Opening File: " << filename << std::endl;
    exit(EXIT_FAILURE);
  }
  std::string line;
  do
  {
    if (!std::getline(GraphInput, line))
      return graph_data;
  } while (IsMetisComment(line));

  int num_v = 0, num_edges = 0, fmt = 0;
  {
    std::istringstream iss(line);
    iss >> num_v >> num_edges >> fmt;
  }

  const bool e_w = (fmt & 1);
  const bool v_w = (fmt & 2);

  graph_data.num_v = num_v;
  graph_data.num_edge = num_edges;

  graph_data.offsets.reserve(num_v + 1);
  graph_data.csr.reserve(num_edges);
  graph_data.s_edge.reserve(num_edges);
  graph_data.offsets.push_back(0);

  int src = 0;
  while (std::getline(GraphInput, line))
  {
    if (IsMetisComment(line) || line.empty())
      continue;

    const char *p = line.c_str();

    if (v_w)
    {
      while (*p == ' ' || *p == '\t')
        ++p;
      while (*p >= '0' && *p <= '9')
        ++p;
    }

    int offset_start = static_cast<int>(graph_data.csr.size());

    while (*p != '\0')
    {
      while (*p == ' ' || *p == '\t')
        ++p;
      if (*p == '\0')
        break;
      int dst = 0;
      while (*p >= '0' && *p <= '9')
        dst = dst * 10 + (*p++ - '0');
      dst -= 1;

      if (e_w)
      {
        while (*p == ' ' || *p == '\t')
          ++p;
        while (*p >= '0' && *p <= '9')
          ++p;
      }

      graph_data.csr.push_back(dst);
      graph_data.s_edge.push_back(src);
    }

    graph_data.offsets.push_back(static_cast<int>(graph_data.csr.size()));
    ++src;
  }

  while (static_cast<int>(graph_data.offsets.size()) <= num_v)
    graph_data.offsets.push_back(graph_data.offsets.back());

  graph_data.num_edge = static_cast<int>(graph_data.csr.size());
  return graph_data;
}
GraphData readGraph_Forward(const std::string &filename)
{
  GraphData graph_data = {0, 0, {}, {}, {}};

  std::ifstream GraphInput(filename);
  if (!GraphInput.is_open())
  {
    std::cerr << "Error Opening File: " << filename << std::endl;
    exit(EXIT_FAILURE);
  }
  std::string line;
  do
  {
    if (!std::getline(GraphInput, line))
      return graph_data;
  } while (IsMetisComment(line));

  int num_v = 0, num_edges = 0, fmt = 0;
  {
    std::istringstream iss(line);
    iss >> num_v >> num_edges >> fmt;
  }

  const bool e_w = (fmt & 1);
  const bool v_w = (fmt & 2);

  graph_data.num_v = num_v;
  graph_data.num_edge = num_edges;

  graph_data.offsets.reserve(num_v + 1);
  graph_data.csr.reserve(num_edges);
  graph_data.s_edge.reserve(num_edges);
  graph_data.offsets.push_back(0);

  int src = 0;
  while (std::getline(GraphInput, line))
  {
    if (IsMetisComment(line) || line.empty())
      continue;

    const char *p = line.c_str();

    if (v_w)
    {
      while (*p == ' ' || *p == '\t')
        ++p;
      while (*p >= '0' && *p <= '9')
        ++p;
    }

    int offset_start = static_cast<int>(graph_data.csr.size());

    while (*p != '\0')
    {
      while (*p == ' ' || *p == '\t')
        ++p;
      if (*p == '\0')
        break;
      int dst = 0;
      while (*p >= '0' && *p <= '9')
        dst = dst * 10 + (*p++ - '0');
      dst -= 1;

      if (e_w)
      {
        while (*p == ' ' || *p == '\t')
          ++p;
        while (*p >= '0' && *p <= '9')
          ++p;
      }

      if (dst <= src)
        continue;

      graph_data.csr.push_back(dst);
      graph_data.s_edge.push_back(src);
    }

    graph_data.offsets.push_back(static_cast<int>(graph_data.csr.size()));
    ++src;
  }

  while (static_cast<int>(graph_data.offsets.size()) <= num_v)
    graph_data.offsets.push_back(graph_data.offsets.back());

  graph_data.num_edge = static_cast<int>(graph_data.csr.size());
  return graph_data;
}
