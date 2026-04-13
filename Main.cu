#include "FileReader/command_args.h"
#include "FileReader/FileReader.h"
#include <string>
int main(int argc, char *argv[])
{
    auto t1 = std::chrono::high_resolution_clock::now();
    CommandArgs ca = parse_command_args(argc, argv);
    GraphFR FR(ca);
    FR.ReadFile();
    int triangles = FR.CalculateTriangles();
    std::cout << "Number of triangles: " << triangles << std::endl;
    auto t2 = std::chrono::high_resolution_clock::now();
    //in milliseconds
    std::chrono::duration<double, std::milli> elapsed = t2 - t1;
    std::cout << "Execution time: " << elapsed.count() << " ms" << std::endl;
    return 0;
}