#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <map>
using namespace std;
struct Edge
{
    int src;
    int dst;
};
 struct Graph
{
    int num_vertices = 0, num_edges = 0;
    map<int, vector<int>> adj_list;
};

/*

Matrix Market (.mtx) to METIS graph format converter.
Designed for large graphs: O(E) memory, streaming output.

MTX format (coordinate):
  %%MatrixMarket matrix coordinate <field> <symmetry>
  % comment lines
  <num_rows> <num_cols> <num_entries>
  <row> <col> [value]

METIS format:
  <num_vertices> <num_edges>
  <adj1> <adj2> ...   (one line per vertex, 1-based, undirected)
*/

int main(int argc, char *argv[])
{
    string input = argv[1], output = argv[2];
    ifstream infile(input);
    ofstream outfile(output);
    if (!infile.is_open())
    {
        cerr << "Error opening file: " << input << endl;
        return 1;
    }
    string line;
    Graph graph;
    cout<<"Reading MTX file: " << input << endl;
    while(getline(infile, line))
    {
        if (line[0] == '%') continue; // Skip comments
        if(line.empty()) continue; // Skip empty lines
        if (graph.num_vertices == 0)
        {
            // First non-comment line: num_rows num_cols num_entries
            istringstream iss(line);
            int num_rows, num_cols, num_entries;
            iss >> num_rows >> num_cols >> num_entries;
            graph.num_vertices = max(num_rows, num_cols);
            graph.num_edges = num_entries;
        }
        else
        {
            // Edge line: row col [value]
            istringstream iss(line);
            int row, col;
            iss >> row >> col; // Ignore value if present
            graph.adj_list[row].push_back(col);
            graph.adj_list[col].push_back(row); // Undirected
        }
    }
    infile.close();
    cout << "Finished reading MTX. Vertices: " << graph.num_vertices << ", Edges: " << graph.num_edges << endl;
    // Write METIS format
    outfile << graph.num_vertices << " " << graph.num_edges << endl;
    for (int v = 1; v <= graph.num_vertices; v++)
    {
        if (graph.adj_list.count(v))
        {
            for (int adj : graph.adj_list[v])
            {
                outfile << adj << " ";
            }
        }
        outfile << endl;
    }
    outfile.close();
    cout << "[+]" << " Finished writing METIS file: " << output << endl;
    return 0;
};