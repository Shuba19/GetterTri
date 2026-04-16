#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <set>
using namespace std;

/*
METIS graph format to Matrix Market (.mtx) converter.

METIS format:
  % comment lines (ignored)
  <num_vertices> <num_edges> [fmt] [ncon]   (header)
  <adj1> <adj2> ...                          (una riga per vertice, 1-based)
  ...

Matrix Market format:
  %%MatrixMarket matrix coordinate pattern symmetric
  <num_rows> <num_cols> <num_entries>   (num_entries = num_edges, upper triangle)
  <row> <col>                           (1-based, row <= col)
*/

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        cerr << "Usage: " << argv[0] << " <input_metis> <output_mtx>" << endl;
        return 1;
    }

    string input = argv[1], output = argv[2];
    ifstream infile(input);

    if (!infile.is_open())
    {
        cerr << "Error opening input file: " << input << endl;
        return 1;
    }

    // Leggi header saltando commenti
    string line;
    int num_vertices = 0, num_edges = 0;
    while (getline(infile, line))
    {
        if (line.empty()) continue;
        if (line[0] == '%') continue; // Commenti METIS

        istringstream iss(line);
        if (!(iss >> num_vertices >> num_edges))
        {
            cerr << "Error parsing METIS header." << endl;
            return 1;
        }
        break; // Header letto, esci
    }

    if (num_vertices == 0)
    {
        cerr << "Empty or invalid METIS file." << endl;
        return 1;
    }

    // Leggi lista adiacenza e deduplicat in upper triangle
    set<pair<int,int>> edges;
    int vertex = 1;
    while (getline(infile, line))
    {
        if (line.empty())
        {
            // Riga vuota = nodo senza archi, avanza comunque
            vertex++;
            continue;
        }
        if (line[0] == '%') continue;

        istringstream iss(line);
        int adj;
        while (iss >> adj)
        {
            if (adj == vertex) continue; // salta self-loop
            // Salva solo upper triangle (row <= col)
            int r = min(vertex, adj);
            int c = max(vertex, adj);
            edges.insert({r, c});
        }
        vertex++;

        if (vertex > num_vertices) break;
    }
    infile.close();

    ofstream outfile(output);
    if (!outfile.is_open())
    {
        cerr << "Error opening output file: " << output << endl;
        return 1;
    }

    outfile << "%%MatrixMarket matrix coordinate pattern symmetric\n";
    outfile << "% Converted from METIS format: " << input << "\n";
    outfile << num_vertices << " " << num_vertices << " " << (int)edges.size() << "\n";

    for (const auto& e : edges)
        outfile << e.first << " " << e.second << "\n";

    outfile.close();
    cout << "Done: " << num_vertices << " vertices, " << edges.size() << " edges." << endl;
    return 0;
}