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

inline int quick_atoi(const char *&p)
{
  int res = 0;
  while (*p >= '0' && *p <= '9')
  {
    res = res * 10 + (*p++ - '0');
  }
  return res;
}

inline void skip_whitespace(const char *&p)
{
  while (*p == ' ' || *p == '\t' || *p == '\r')
    ++p;
}

GraphData readGraph(const std::string &filename)
{
  int fd = open(filename.c_str(), O_RDONLY);
  if (fd == -1)
  {
    std::cerr << "Errore apertura file" << std::endl;
    exit(EXIT_FAILURE);
  }

  struct stat sb;
  fstat(fd, &sb);
  const char *file_ptr = static_cast<const char *>(mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0));
  const char *end_ptr = file_ptr + sb.st_size;

  GraphData graph_data;
  const char *p = file_ptr;

  // 1. Salta i commenti iniziali
  while (p < end_ptr && *p == '%')
  {
    while (p < end_ptr && *p != '\n')
      ++p;
    if (p < end_ptr)
      ++p;
  }

  // 2. Leggi Header (V, E, FMT)
  skip_whitespace(p);
  int num_v = quick_atoi(p);
  skip_whitespace(p);
  int num_e_meta = quick_atoi(p); // Num archi dichiarati
  skip_whitespace(p);
  int fmt = (*p >= '0' && *p <= '9') ? quick_atoi(p) : 0;
  while (p < end_ptr && *p != '\n')
    ++p;
  if (p < end_ptr)
    ++p;

  const bool e_w = (fmt & 1);
  const bool v_w = (fmt & 2);
  graph_data.num_v = num_v;
  graph_data.offsets.reserve(num_v + 1);
  graph_data.csr.reserve(num_e_meta);
  graph_data.s_edge.reserve(num_e_meta);
  graph_data.offsets.push_back(0);

  int src = 0;
  while (p < end_ptr && src < num_v)
  {
    // Skip comment rows anywhere in the file.
    if (*p == '%')
    {
      while (p < end_ptr && *p != '\n')
        ++p;
      if (p < end_ptr)
        ++p;
      continue;
    }

    // A blank row is a valid isolated vertex in METIS.
    if (*p == '\n')
    {
      graph_data.offsets.push_back(static_cast<int>(graph_data.csr.size()));
      ++src;
      ++p;
      continue;
    }

    skip_whitespace(p);
    if (v_w)
    {
      quick_atoi(p);
      skip_whitespace(p);
    }

    while (p < end_ptr && *p != '\n')
    {
      int dst = quick_atoi(p) - 1;

      if (e_w)
      {
        skip_whitespace(p);
        quick_atoi(p);
      }

      graph_data.csr.push_back(dst);
      graph_data.s_edge.push_back(src);

      skip_whitespace(p);
    }

    graph_data.offsets.push_back(static_cast<int>(graph_data.csr.size()));
    if (p < end_ptr)
      ++p;
    ++src;
  }

  while (static_cast<int>(graph_data.offsets.size()) <= num_v)
  {
    graph_data.offsets.push_back(graph_data.offsets.back());
  }

  graph_data.num_edge = static_cast<int>(graph_data.csr.size());

  munmap((void *)file_ptr, sb.st_size);
  close(fd);

  return graph_data;
}

GraphData readGraph_Forward(const std::string &filename)
{
  int fd = open(filename.c_str(), O_RDONLY);
  if (fd == -1)
  {
    std::cerr << "Errore apertura file" << std::endl;
    exit(EXIT_FAILURE);
  }
  struct stat sb;
  fstat(fd, &sb);
  const char *file_ptr = static_cast<const char *>(mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0));
  const char *end_ptr = file_ptr + sb.st_size;

  GraphData graph_data;
  const char *p = file_ptr;
  
  // Skip commenti iniziali
  while (p < end_ptr && *p == '%')
  {
    while (p < end_ptr && *p != '\n') ++p;
    if (p < end_ptr) ++p;
  }

  skip_whitespace(p);
  int num_v = quick_atoi(p);
  skip_whitespace(p);
  int num_e_meta = quick_atoi(p);
  skip_whitespace(p);
  int fmt = (*p >= '0' && *p <= '9') ? quick_atoi(p) : 0;
  
  while (p < end_ptr && *p != '\n') ++p;
  if (p < end_ptr) ++p;

  const bool e_w = (fmt & 1);
  const bool v_w = (fmt & 2);

  graph_data.num_v = num_v;
  graph_data.offsets.reserve(num_v + 1);
  graph_data.csr.reserve(num_e_meta); 
  graph_data.s_edge.reserve(num_e_meta);
  graph_data.offsets.push_back(0);
  
  std::vector<int> row_neighbors; 
  row_neighbors.reserve(1024);    

  int src = 0;
  while (p < end_ptr && src < num_v)
  {
    if (*p == '%')
    {
      while (p < end_ptr && *p != '\n') ++p;
      if (p < end_ptr) ++p;
      continue;
    }

    if (*p == '\n')
    {
      graph_data.offsets.push_back(static_cast<int>(graph_data.csr.size()));
      ++src;
      ++p;
      continue;
    }

    skip_whitespace(p);
    if (v_w)
    {
      quick_atoi(p);
      skip_whitespace(p);
    }

    row_neighbors.clear(); 
    while (p < end_ptr && *p != '\n')
    {
      int dst = quick_atoi(p) - 1;

      if (e_w)
      {
        skip_whitespace(p);
        quick_atoi(p);
      }
      
      if (dst < src)
      {
        row_neighbors.push_back(dst);
      }
      skip_whitespace(p);
    }

    if (!row_neighbors.empty()) {
        std::sort(row_neighbors.begin(), row_neighbors.end());
        for (int neighbor : row_neighbors) {
            graph_data.csr.push_back(neighbor);
            graph_data.s_edge.push_back(src);
        }
    }

    graph_data.offsets.push_back(static_cast<int>(graph_data.csr.size()));
    if (p < end_ptr) ++p;
    ++src;
  }

  while (static_cast<int>(graph_data.offsets.size()) <= num_v)
  {
    graph_data.offsets.push_back(graph_data.offsets.back());
  }

  graph_data.num_edge = static_cast<int>(graph_data.csr.size());

  munmap((void *)file_ptr, sb.st_size);
  close(fd);

  return graph_data;
}

void print_output_as_json(const output_t &output)
{
  std::string name = output.file;
  size_t last_slash = output.file.find_last_of("/\\");
  if (last_slash != std::string::npos)
  {
    name = name.substr(last_slash + 1);
  }
  size_t last_dot = name.find_last_of(".");
  if (last_dot != std::string::npos)
  {
    name = name.substr(0, last_dot);
  }
  auto density = (output.num_v > 1) ? (2.0 * output.num_e) / (output.num_v * (output.num_v - 1)) : 0.0;
  std::cout << "{\n";
  std::cout << "  \"file\": \"" << name << "\",\n";
  std::cout << "  \"num_vertices\": " << output.num_v << ",\n";
  std::cout << "  \"num_edges\": " << output.num_e << ",\n";
  std::cout << "  \"density\": " << density << ",\n";

  std::cout << "  \"triangles\": " << output.triangles << ",\n";
  std::cout << "  \"total_space\": " << output.memory_total << ",\n";
  std::cout << "  \"unit_memory\": \"" << output.unit_memory << "\",\n";
  std::cout << "  \"total_time\": " << output.total_time << ",\n";
  std::cout << "  \"read_time\": " << output.read_time << ",\n";
  std::cout << "  \"preprocess_time\": " << output.preprocess_time << ",\n";
  std::cout << "  \"kernel_time\": " << output.kernel_time << ",\n";
  std::cout << "  \"unit_time\": \"" << output.unit_time << "\",\n";
  std::cout << "  \"time_per_threshold\": [\n";
  for (auto &entry : output.time_per_threshold)
  {
    int threshold = entry.first;
    std::cout << "    {\n";
    std::cout << "      \"threshold\": " << threshold << ",\n";
    std::cout << "      \"blocks\": [\n";
    for (auto &block_entry : entry.second)
    {
      int block_size = block_entry.first;
      float time = block_entry.second;
      std::cout << "        {";
      std::cout << "\"block_size\": " << block_size << ",";
      std::cout << "\"time\": " << time;
      std::cout << "}";
      if (block_entry != *entry.second.rbegin())
        std::cout << ",";
      std::cout << "\n";
    }
    std::cout << "      ]\n";
    std::cout << "    }\n";
    if (entry != *output.time_per_threshold.rbegin())
      std::cout << ",\n";
  }
  std::cout << "\n  ]\n";
  std::cout << "}\n";
}