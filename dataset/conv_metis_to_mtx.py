import sys

def convert_metis_to_mtx(metis_file, mtx_file):
    print(f"Reading {metis_file}...")
    
    with open(metis_file, 'r') as f:
        # Read header
        line = f.readline()
        parts = line.split()
        num_vertices = int(parts[0])
        num_edges = int(parts[1])
        
        print(f"Graph dimensions: {num_vertices} vertices, {num_edges} edges")
        
        # Prepare adjacency list
        adj = [[] for _ in range(num_vertices + 1)]
        
        for i in range(1, num_vertices + 1):
            line = f.readline().strip()
            if line:
                neighbors = list(map(int, line.split()))
                adj[i].extend(neighbors)
    
    print(f"Writing {mtx_file}...")
    
    with open(mtx_file, 'w') as f:
        # Write MTX header
        f.write("%%MatrixMarket matrix coordinate pattern symmetric\n")
        f.write(f"{num_vertices} {num_vertices} {num_edges}\n")
        
        # Write edges (u,v) for each neighbor v of u
        for u in range(1, num_vertices + 1):
            for v in adj[u]:
                if u < v:  # To avoid duplicates in symmetric format
                    f.write(f"{v} {u}\n")

    print("Conversion complete.")
    
    
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_metis_to_mtx.py <input.metis> <output.mtx>")
        sys.exit(1)
    
    metis_file = sys.argv[1]
    mtx_file = sys.argv[2]
    
    convert_metis_to_mtx(metis_file, mtx_file)