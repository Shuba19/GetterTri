#include "FileReader.h"

#define LIMIT 16
#define DEGORDER false
/*********************************
 * Metis Graph first line parameters:
 *
 * - V    : number of vertexes
 * - E    : number of edges
 * - fmt  : can have four values (0 no weight, 1: edges weighted, 10: vertices weighted, 11: edges and vertices weighted)
 * - ncon : number of weight associated to vertices
 /*******************************/

#define REP_BENCHMARK 50
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
  int corrector = this->args.corrector ? 1 : 0;
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
#if !DEGORDER
      if (edge - corrector <= pos && this->args.mode != 2 && this->args.mode != 5 && this->args.mode != 6)
        continue;
      int mode = this->args.mode;
      int lim = ((pos/16) +1) * 16;
      if ((mode == 4 || mode == 5 || mode == 6) && edge - corrector > lim)
        continue; 
#endif
      v_edge.push_back(edge - corrector);

      if (e_weight)
        iss >> garbage;
    }
    sum = sum + v_edge.size();
    offsets.push_back(sum);
    if (!v_edge.empty())
    {
      for (auto i : v_edge)
      {
        csr.push_back(i);
        s_edge.push_back(pos);
      }
    }
    pos++;
    if (this->args.mode == 4)
      filter_per_deg(offsets, s_edge, this->th_level, this->warp_level);
  }
  this->num_edge = this->num_edge << 1;
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

  if (this->args.snap)
  {
    bool result = SNAP_Reader(GraphInput, false, false, 0);
    GraphInput.close();
    return result;
  }
  if (this->args.mode == 7)
  {
    bool result = Tile_Reader(GraphInput, false, false, 0);
    GraphInput.close();
    return result;
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
#if DEGORDER
  ReorderByDeg();
#endif

  return result;
}

bool GraphFR::ReorderByDeg()
{
  std::vector<std::pair<int, int>> deg_node(num_v);

  for (int i = 0; i < num_v; i++)
  {
    deg_node[i] = {offsets[i + 1] - offsets[i], i};
  }
  // 2. Ordina per grado crescente
  std::sort(deg_node.begin(), deg_node.end());
  // 3. Crea mappa da vecchio a nuovo ID
  std::vector<int> old_to_new(num_v);
  for (int i = 0; i < num_v; i++)
  {
    old_to_new[deg_node[i].second] = i;
  }
  // 4. Ricostruisci CSR con nuovi ID mantenendo ogni lista ordinata
  std::vector<int> final_csr;
  std::vector<int> final_s_edge;
  std::vector<int> final_offsets(num_v + 1, 0);
  final_csr.reserve(csr.size());
  final_s_edge.reserve(s_edge.size());

  for (int i = 0; i < num_v; i++)
  {
    int old_id = deg_node[i].second;

    std::vector<int> mapped_neighbors;
    mapped_neighbors.reserve(offsets[old_id + 1] - offsets[old_id]);
    for (int j = offsets[old_id]; j < offsets[old_id + 1]; j++)
    {
      int old_neighbor = csr[j];
      mapped_neighbors.push_back(old_to_new[old_neighbor]);
    }

    std::sort(mapped_neighbors.begin(), mapped_neighbors.end());

    final_offsets[i + 1] = final_offsets[i];
    for (int neighbor : mapped_neighbors)
    {
      if (this->args.mode != 2 && this->args.mode != 5 && this->args.mode != 6 && neighbor < i)
        continue;
      final_csr.push_back(neighbor);
      final_s_edge.push_back(i);
      final_offsets[i + 1]++;
    }
  }

  this->csr = std::move(final_csr);
  this->s_edge = std::move(final_s_edge);
  this->offsets = std::move(final_offsets);
  this->num_edge = static_cast<int>(this->csr.size());

  if (this->args.mode == 4)
  {
    this->th_level.clear();
    this->warp_level.clear();
    filter_per_deg(this->offsets, this->s_edge, this->th_level, this->warp_level);
  }

  return true;
}
bool GraphFR::SNAP_Reader(std::ifstream &GraphInput, bool e_weight, bool v_weight, int n_skip)
{
  (void)e_weight;
  (void)v_weight;
  (void)n_skip;

  this->csr.clear();
  this->s_edge.clear();
  this->offsets.clear();

  std::map<int, std::vector<int>> adjacency_list;
  std::string line;
  this->num_v = 0;
  this->num_edge = 0;
  int input_corrector = this->args.corrector ? 1 : 0;
  int max_vertex_id = -1;
  while (std::getline(GraphInput, line))
  {
    if (line.empty() || line[0] == '#')
      continue;
    std::istringstream iss(line);
    int src, dst;
    if (!(iss >> src >> dst))
      continue;

    int s = src - input_corrector, d = dst - input_corrector;
    if (s < 0 || d < 0 || s == d)
      continue;

    adjacency_list[s].push_back(d);
    adjacency_list[d].push_back(s);
    if (s > max_vertex_id)
      max_vertex_id = s;
    if (d > max_vertex_id)
      max_vertex_id = d;
    this->num_edge++;
  }

  if (adjacency_list.empty())
  {
    this->offsets.push_back(0);
    return true;
  }

  this->num_v = max_vertex_id + 1;

  this->offsets.reserve(this->num_v + 1);
  this->offsets.push_back(0);
  for (int v = 0; v < this->num_v; ++v)
  {
    auto it = adjacency_list.find(v);
    if (it != adjacency_list.end())
    {
      std::vector<int> &neighbors = it->second;
      std::sort(neighbors.begin(), neighbors.end());
      neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());

      for (int dst : neighbors)
      {
        if (this->args.mode != 2 && this->args.mode != 5  &&  this->args.mode != 6 && dst < v)
          continue;
        
        this->csr.push_back(dst);
        this->s_edge.push_back(v);
      }
    }
    this->offsets.push_back(this->csr.size());
  }

  this->num_edge = static_cast<int>(this->csr.size());
  if (this->args.mode == 4)
    filter_per_deg(offsets, s_edge, this->th_level, this->warp_level);

  return true;
}
int sum_tiles(const tiles_b &tile)
{
  int sum = 0;
  for (int i = 0; i < 16; i++)
    sum += tile.tile[i];
  return sum;
}
bool GraphFR::Tile_Reader(std::ifstream &GraphInput, bool e_weight, bool v_weight, int n_skip)
{
  this->tiles.clear();
  this->valid_tile.clear();

  // Leggi tutto il file in memoria in una volta sola
  std::ostringstream buffer;
  buffer << GraphInput.rdbuf();
  std::string content = buffer.str();
  const char *ptr = content.c_str();

  std::vector<tiles_b> temp_tiles;
  temp_tiles.reserve(1024);
  valid_tile.reserve(1024);

  int t_line = 0;
  int track_tile = 0;
  int pos = 0;

  /*
   *Ogni 16 righe del file creo un tot di tiles
   * alla i_esima iterazione, verranno generate i+1 tiles provvisori, poi scartati se  non validi
   * verranno salvati solo i tile che non sono vuoti
   *
   */
  // la prima riga del file conitene il numero di vertici e di edge
  std::istringstream iss(content.substr(0, content.find('\n')));
  int num_v, num_edges;
  iss >> num_v >> num_edges;
  this->num_v = num_v;
  this->num_edge = num_edges;

  while (*ptr)
  {
    // Raccogli fino a 16 righe valide
    std::vector<const char *> line_starts;
    std::vector<const char *> line_ends;
    line_starts.reserve(16);
    line_ends.reserve(16);

    while (t_line < 16 && *ptr)
    {
      const char *line_start = ptr;

      // Avanza fino a fine riga
      while (*ptr && *ptr != '\n')
        ptr++;
      const char *line_end = ptr;
      if (*ptr == '\n')
        ptr++;

      // Salta righe vuote o commenti
      if (line_start == line_end || *line_start == '#')
        continue;

      line_starts.push_back(line_start);
      line_ends.push_back(line_end);
      t_line++;
    }

    if (line_starts.empty())
      break;

    std::vector<tiles_b> row_tiles(pos + 1);
    int limit = 16 * (pos + 1);

    for (int row = 0; row < (int)line_starts.size(); row++)
    {
      // qua ogni riga viene letta, viene preso al max 16*(pos+1) valori:
      // EX : pos = 0 -> 16 valori, pos = 1 -> 32 valori, pos = 2 -> 48 valori ect...
      // se esiste un edge viene pushato un 1 altrimenti niente
      const char *lptr = line_starts[row];
      const char *lend = line_ends[row];

      while (lptr < lend)
      {
        // Salta spazi
        while (lptr < lend && (*lptr == ' ' || *lptr == '\t'))
          lptr++;
        if (lptr >= lend)
          break;

        // Parsing manuale dell'intero
        char *end;
        int edge = (int)std::strtol(lptr, &end, 10);
        if (end == lptr)
          break;
        lptr = end;

        if (edge > limit)
          break;

        int tile_index = (edge - 1) / 16;
        if (tile_index > pos)
          break;

        // i tiles sono composti da 16 uint16_t, se esiste l'arco allora il row-esimo bit viene settato a 1
        row_tiles[tile_index].tile[row] |= (1 << ((edge - 1) % 16));
      }
    }

    for (const tiles_b &tile : row_tiles)
    {
      if (sum_tiles(tile) > 0)
      {
        temp_tiles.push_back(tile);
        valid_tile.push_back(track_tile);
      }
      track_tile++;
    }

    t_line = 0;
    pos++;
  }

  this->tiles = temp_tiles;
  std::cout << "Number of valid tiles: " << this->tiles.size() << std::endl;
  return true;
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
  std::string mode = "";
  if (this->args.mode == 0)
    mode = "Edge Iterator";
  else if (this->args.mode == 1)
    mode = "Node Iterator";
  else if (this->args.mode == 2)
    mode = "Tensor Calculation";
  else if (this->args.mode == 3)
    mode = "OpenMP CPU Calculation";
  else if (this->args.mode == 4)
    mode = "Adaptive Edge Search";
  else if (this->args.mode == 5)
    mode = "TTC_2";
  else if (this->args.mode == 6)
    mode = "TTC_3";
  else if (this->args.mode == 7)
    mode = "Tile Reader";
  else
    mode = "Unknown";

  std::cout << "----------------------------------" << std::endl;
  std::cout << "Graph Information:" << std::endl;
  std::cout << "Number of vertices: " << this->num_v << std::endl;
  std::cout << "Number of edges: " << this->num_edge << std::endl;
  std::cout << "Graph Density: " << (density * 100.0) << "%" << std::endl;
  std::cout << "CSR Size: " << this->csr.size() << std::endl;
  std::cout << "Offsets Size: " << this->offsets.size() << std::endl;
  std::cout << "Mode: " << mode << std::endl;
  if (this->args.benchmark)
  {
    std::cout << "Benchmarking mode enabled with " << REP_BENCHMARK << " repetitions." << std::endl;
    std::cout << "Time: " << this->timer.time / REP_BENCHMARK << " ms" << std::endl;
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
    triangle_count = SearchTriangle_Edge_Iterator(this->num_v, this->num_edge, this->offsets, this->csr, this->s_edge);
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
  case 4:
    triangle_count = adaptive_edge_search(this->num_v, this->num_edge, this->offsets, this->csr, this->s_edge, this->th_level, this->warp_level);
    break;
  case 5:
    triangle_count = TTC_2(this->num_v, this->num_edge, this->offsets, this->csr);
    break;
  case 6:
    triangle_count = TTC_3(this->num_v, this->num_edge, this->offsets, this->csr);
    break;
  case 7:
    triangle_count = TTC_4(this->num_v, this->num_edge, this->tiles, this->valid_tile);

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
  SearchTriangle_Edge_Iterator(this->num_v, this->num_edge, this->offsets, this->csr, this->s_edge);
  StartTimer();

  switch (this->args.mode)
  {
  case 0:
    for (int i = 0; i < REP_BENCHMARK; i++)
      SearchTriangle_Edge_Iterator(this->num_v, this->num_edge, this->offsets, this->csr, this->s_edge);
    break;
  case 1:
    for (int i = 0; i < REP_BENCHMARK; i++)
      SearchTriangle_Node_Iterator(this->num_v, this->num_edge, this->offsets, this->csr, this->args.undirect);
    break;
  case 2:
    for (int i = 0; i < REP_BENCHMARK; i++)
      TTC(this->num_v, this->num_edge, this->offsets, this->csr);
    break;
  case 3:
    for (int i = 0; i < REP_BENCHMARK; i++)
      triangle_couting_CPU(this->num_v, this->num_edge, this->offsets, this->csr);
    break;
  case 4:
    for (int i = 0; i < REP_BENCHMARK; i++)
      adaptive_edge_search(this->num_v, this->num_edge, this->offsets, this->csr, this->s_edge, this->th_level, this->warp_level);
    break;
  case 5:
    for (int i = 0; i < REP_BENCHMARK; i++)
      TTC_2(this->num_v, this->num_edge, this->offsets, this->csr);
    break;
  case 6:
    for (int i = 0; i < REP_BENCHMARK; i++)
      TTC_3(this->num_v, this->num_edge, this->offsets, this->csr);
    break;
  case 7:
    // triangle_count = TTC_4(this->num_v, this->num_edge, this->tiles, this->valid_tile);

    break;
  default:
    std::cerr << "Invalid mode selected. Please choose 'n' for Node Iterator, 'e' for Edge Iterator, or 't' for Tensor Calculation." << std::endl;
    return;
  }
  StopTimer();
  printVerboseGraphInfo();
  std::cout << "------- END OF BENCHMARK ---------" << std::endl;
}
