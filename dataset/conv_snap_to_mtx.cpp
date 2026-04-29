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

/*
SNAP edge-list to Matrix Market (.mtx) format converter.

SNAP format:
  # comment lines (ignored)
  <from_node_id> <to_node_id>   (whitespace separated, 0- or 1-based)
  ...

Matrix Market format:
  %%MatrixMarket matrix coordinate pattern symmetric
  % optional comments
  <num_rows> <num_cols> <num_entries>   (num_entries = num_edges, upper triangle only for symmetric)
  <row> <col>                           (1-based, only upper triangle: row <= col)

Node IDs in SNAP sono rimappati a range 1-based contiguo.
*/

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        cerr << "Usage: " << argv[0] << " <input_snap> <output_mtx>" << endl;
        return 1;
    }

    string input = argv[1], output = argv[2];
    ifstream infile(input);

    if (!infile.is_open())
    {
        cerr << "Error opening input file: " << input << endl;
        return 1;
    }

    string line;
    set<pair<int,int>> edges;
    set<int> nodes;

    while (getline(infile, line))
    {
        if (line.empty()) continue;
        if (line[0] == '#') continue; // Commenti SNAP

        istringstream iss(line);
        int row, col;
        if (!(iss >> row >> col)) continue; // Riga malformata: salta

        if (row == col) continue; // Salta self-loop

        nodes.insert(row);
        nodes.insert(col);

        if (row > col) swap(row, col);
        edges.insert({row, col});
    }
    infile.close();

    // Remapping nodi a range 1-based contiguo
    map<int, int> id_remap;
    int new_id = 1;
    for (int node : nodes)
    {
        id_remap[node] = new_id++;
    }

    int num_vertices = (int)nodes.size();
    int num_edges    = (int)edges.size();

    ofstream outfile(output);
    if (!outfile.is_open())
    {
        cerr << "Error opening output file: " << output << endl;
        return 1;
    }

    // Header Matrix Market
    outfile << "%%MatrixMarket matrix coordinate pattern symmetric\n";
    outfile << "% Converted from SNAP format: " << input << "\n";
    outfile << num_vertices << " " << num_vertices << " " << num_edges << "\n";

    // Scrivi archi (upper triangle, 1-based, rimappati)
    for (const auto& e : edges)
    {
        outfile << id_remap[e.first] << " " << id_remap[e.second] << "\n";
    }

    outfile.close();
    cout << "Done: " << num_vertices << " vertices, " << num_edges << " edges." << endl;
    return 0;
}