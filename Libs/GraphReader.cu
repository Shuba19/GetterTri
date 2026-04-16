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

  // Pre-allocazione massiva
  graph_data.num_v = num_v;
  graph_data.offsets.reserve(num_v + 1);
  graph_data.csr.reserve(num_e_meta); // Nota: potrebbe essere sovrastimato se dst <= src
  graph_data.s_edge.reserve(num_e_meta);
  graph_data.offsets.push_back(0);

  int src = 0;
  while (p < end_ptr && src < num_v)
  {
    // Salta commenti tra le righe o righe vuote
    if (*p == '%' || *p == '\n')
    {
      if (*p == '%')
        while (p < end_ptr && *p != '\n')
          ++p;
      if (p < end_ptr)
        ++p;
      continue;
    }

    // Salta peso del vertice se presente
    skip_whitespace(p);
    if (v_w)
    {
      quick_atoi(p);
      skip_whitespace(p);
    }

    while (p < end_ptr && *p != '\n')
    {
      int dst = quick_atoi(p) - 1;

      // Salta peso dell'arco se presente
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
      ++p; // Vai alla riga successiva
    ++src;
  }

  // Gestione vertici isolati finali
  while (static_cast<int>(graph_data.offsets.size()) <= num_v)
  {
    graph_data.offsets.push_back(graph_data.offsets.back());
  }

  graph_data.num_edge = static_cast<int>(graph_data.csr.size());

  // Cleanup
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

  // Pre-allocazione massiva
  graph_data.num_v = num_v;
  graph_data.offsets.reserve(num_v + 1);
  graph_data.csr.reserve(num_e_meta); // Nota: potrebbe essere sovrastimato se dst <= src
  graph_data.s_edge.reserve(num_e_meta);
  graph_data.offsets.push_back(0);

  int src = 0;
  while (p < end_ptr && src < num_v)
  {
    // Salta commenti tra le righe o righe vuote
    if (*p == '%' || *p == '\n')
    {
      if (*p == '%')
        while (p < end_ptr && *p != '\n')
          ++p;
      if (p < end_ptr)
        ++p;
      continue;
    }

    // Salta peso del vertice se presente
    skip_whitespace(p);
    if (v_w)
    {
      quick_atoi(p);
      skip_whitespace(p);
    }

    while (p < end_ptr && *p != '\n')
    {
      int dst = quick_atoi(p) - 1;

      // Salta peso dell'arco se presente
      if (e_w)
      {
        skip_whitespace(p);
        quick_atoi(p);
      }

      if (dst > src)
      {
        graph_data.csr.push_back(dst);
        graph_data.s_edge.push_back(src);
      }
      skip_whitespace(p);
    }

    graph_data.offsets.push_back(static_cast<int>(graph_data.csr.size()));
    if (p < end_ptr)
      ++p; // Vai alla riga successiva
    ++src;
  }

  // Gestione vertici isolati finali
  while (static_cast<int>(graph_data.offsets.size()) <= num_v)
  {
    graph_data.offsets.push_back(graph_data.offsets.back());
  }

  graph_data.num_edge = static_cast<int>(graph_data.csr.size());

  // Cleanup
  munmap((void *)file_ptr, sb.st_size);
  close(fd);

  return graph_data;
}

