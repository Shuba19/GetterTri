#ifndef TILEREADER
#define TILEREADER
#include "TensorUtilities.h"
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <map>
#include <set>
#include <thread>
#include <atomic>
#include <numeric>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
struct TILES{
    int num_v, num_e;
    std::vector<tiles_b> tiles;
    std::vector<int64_t> tile_ids;
};

TILES readTiles(const std::string& filename);

#endif