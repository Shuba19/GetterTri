#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <map>
#include <set>
using namespace std;

struct Graph
{
    int num_vertices = 0, num_edges = 0;
    map<int, set<int>> adj_list; // set<int> evita archi duplicati automaticamente
};

/*
SNAP edge-list to METIS graph format converter.
Designed for large graphs: O(V+E) memory, streaming output.
SNAP format:
  # comment lines (ignored)
  <from_node_id> <to_node_id>   (whitespace separated, 0- or 1-based)
  ...
METIS format:
  <num_vertices> <num_edges>
  <adj1> <adj2> ...   (one line per vertex, 1-based, no self-loops)
Node IDs in SNAP are remapped to a compact 1-based range.
*/

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        cerr << "Usage: " << argv[0] << " <input_snap> <output_metis>" << endl;
        return 1;
    }

    string input = argv[1], output = argv[2];
    ifstream infile(input);
    ofstream outfile(output);

    if (!infile.is_open())
    {
        cerr << "Error opening file: " << input << endl;
        return 1;
    }
    if (!outfile.is_open())
    {
        cerr << "Error opening output file: " << output << endl;
        return 1;
    }

    string line;
    Graph graph;

    while (getline(infile, line))
    {
        if (line.empty()) continue;
        if (line[0] == '#') continue;

        istringstream iss(line);
        int row, col;
        if (!(iss >> row >> col)) continue; // Riga malformata: salta

        if (row == col) continue;

        graph.adj_list[row].insert(col);
        graph.adj_list[col].insert(row); // Grafo non orientato
    }
    infile.close();

    map<int, int> id_remap;
    int new_id = 1;
    for (const auto& kv : graph.adj_list)
    {
        id_remap[kv.first] = new_id++;
    }

    graph.num_vertices = (int)graph.adj_list.size();
    graph.num_edges = 0;
    for (const auto& kv : graph.adj_list)
    {
        graph.num_edges += (int)kv.second.size();
    }
    graph.num_edges /= 2; // Ogni arco contato due volte nel grafo non orientato

    outfile << graph.num_vertices << " " << graph.num_edges << "\n";

    for (const auto& kv : graph.adj_list)
    {
        for (int adj : kv.second)
        {
            outfile << id_remap[adj] << " ";
        }
        outfile << "\n";
    }

    outfile.close();
    return 0;
}