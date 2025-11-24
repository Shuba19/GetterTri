#include "FileReader/command_args.h"
#include "FileReader/FileReader.h"
#include <string>
int main(int argc, char *argv[])
{
    std::string test_path = "../test";
    CommandArgs ca = parse_command_args(argc, argv);
    GraphFR FR(ca);
    FR.ReadFile();
    std::cout << "Counted " << FR.CalculateTriangles()<< " triangles." << std::endl;
    return 0;
}