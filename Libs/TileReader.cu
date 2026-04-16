#include "TileReader.h"
#include <fstream>
#include <iostream>

TILES readTiles(const std::string &filename)
{
    // Apri in modalità binaria
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "Errore apertura file binario: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    TILES tiles;
    
    // 1. Leggi l'header: num_v, num_e, e numero di tile
    file.read(reinterpret_cast<char*>(&tiles.num_v), sizeof(int));
    file.read(reinterpret_cast<char*>(&tiles.num_e), sizeof(int));
    
    uint64_t num_valid_tiles = 0;
    file.read(reinterpret_cast<char*>(&num_valid_tiles), sizeof(uint64_t));

    if (num_valid_tiles == 0)
    {
        std::cerr << "Attenzione: 0 tile nel file." << std::endl;
        file.close();
        return tiles;
    }

    // 2. Alloca la memoria necessaria (molto più veloce del push_back dinamico)
    tiles.tiles.resize(num_valid_tiles);
    tiles.tile_ids.resize(num_valid_tiles);

    // 3. Leggi i dati sequenzialmente
    for (uint64_t i = 0; i < num_valid_tiles; ++i)
    {
        // Legge l'ID del tile (assumendo sia uint64_t)
        file.read(reinterpret_cast<char*>(&tiles.tile_ids[i]), sizeof(uint64_t));
        
        // Legge i 16 bit * 16 righe (32 byte)
        file.read(reinterpret_cast<char*>(tiles.tiles[i].tile), 16 * sizeof(u_int16_t));
    }

    file.close();
    return tiles;
}