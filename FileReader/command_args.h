#include <string>
#include <iostream>
#ifndef FILEREADER_COMMAND_ARGS_HPP
#define FILEREADER_COMMAND_ARGS_HPP

struct CommandArgs {
    std::string input_file;
    int mode = -1;
    bool valid = true;
    bool help = false;
    bool undirect = true;
    bool benchmark = false;
    bool timer = false;
    bool mode_set = false;
    bool verbose = false;
    bool snap = false;
    int corrector = 1;
};
void print_help_message();
CommandArgs parse_command_args(int argc, char** argv);
#endif