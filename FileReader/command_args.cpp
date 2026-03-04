#include "command_args.h"

void print_help_message()
{
    std::cout << "Usage: app -i <input_file> [options]\n"
              << "Options:\n"
              << "  -i <input_file>   Specify the input file\n"
              << "  -d                Use directed graph (default is undirected)\n"
              << "  -b                Enable benchmarking mode\n"
              << "  -h, -help        Display this help message\n"
              << "  -mode             Set the mode (0:Edge Iterator, 1: Node Iterator, 2: Tensor, 3 OpenMP CPU )\n"
              << "  -v          Enable verbose output\n"
              << "  Example:   ./out -i g1.graph -mode 2 -v -b"
              << std::endl;
}


CommandArgs parse_command_args(int argc, char** argv)
{
    CommandArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-i" && i + 1 < argc) {
            args.input_file = argv[++i];
        } else if (arg == "-d") {
            args.undirect = false; 
        }
        else if (arg == "-b") {
            args.benchmark = true; 
        }
        else if (arg == "-h" || arg == "-help") {
            print_help_message();
            exit(0);
        }
        else if (arg == "-mode") {
            args.mode_set = true; 
            args.mode = std::stoi(argv[++i]);
        }
        else if (arg == "-v") {
            args.verbose = true; 
        }
        else if( arg == "-snap")
        {
            args.snap = true;
        }
        else if(arg == "-nc")
        {
            args.corrector = 0;
        }
        else {
            args.valid = false; 
        }
    }
    return args;
}
