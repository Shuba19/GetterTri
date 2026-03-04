#include "FileReader/command_args.h"
#include "FileReader/FileReader.h"
#include <string>
int main(int argc, char *argv[])
{
    CommandArgs ca = parse_command_args(argc, argv);
    GraphFR FR(ca);
    FR.ReadFile();
    int triangles = FR.CalculateTriangles();
    std::cout << "Number of triangles: " << triangles << std::endl;
    return 0;
}