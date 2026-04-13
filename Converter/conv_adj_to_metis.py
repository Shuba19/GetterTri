import sys


def _parse_ints(line):
    return [int(x) for x in line.split()]


def _read_matrix(adj_file):
    with open(adj_file, 'r') as f:
        raw_lines = [line.strip() for line in f if line.strip()]

    if not raw_lines:
        raise ValueError("Input file is empty.")

    first = _parse_ints(raw_lines[0])

    # Case 1: first line is header: "num_vertices num_edges"
    if len(first) == 2 and len(raw_lines) > 1:
        num_vertices = first[0]
        flat_values = []
        for line in raw_lines[1:]:
            flat_values.extend(_parse_ints(line))

        expected = num_vertices * num_vertices
        if len(flat_values) < expected:
            raise ValueError(
                f"Not enough matrix values: expected {expected}, found {len(flat_values)}."
            )

        flat_values = flat_values[:expected]
        matrix = [
            flat_values[i * num_vertices : (i + 1) * num_vertices]
            for i in range(num_vertices)
        ]
        return matrix

    # Case 2: no header, each non-empty line is a matrix row
    matrix = [_parse_ints(line) for line in raw_lines]
    num_vertices = len(matrix)
    for idx, row in enumerate(matrix, start=1):
        if len(row) != num_vertices:
            raise ValueError(
                f"Row {idx} has {len(row)} values, but expected {num_vertices}."
            )

    return matrix


def _matrix_to_adj_list(matrix):
    num_vertices = len(matrix)
    adj = []
    for i in range(num_vertices):
        row = matrix[i]
        neighbors = []
        for j, val in enumerate(row):
            if val == 1 and j != i:
                neighbors.append(j + 1)  # METIS uses 1-based vertex ids
        adj.append(neighbors)
    return adj


def convert_adj_to_metis(adj_file, metis_file):
    print(f"Reading {adj_file}...")
    matrix = _read_matrix(adj_file)
    num_vertices = len(matrix)
    adj = _matrix_to_adj_list(matrix)
    num_edges = sum(len(neighbors) for neighbors in adj) // 2

    print(f"Graph dimensions: {num_vertices} vertices, {num_edges} edges")
    
    print(f"Writing {metis_file}...")
    
    with open(metis_file, 'w') as f:
        # Write METIS header
        f.write(f"{num_vertices} {num_edges}\n")
        
        # Write adjacency list for each vertex
        for neighbors in adj:
            neighbors_str = ' '.join(map(str, neighbors))
            f.write(f"{neighbors_str}\n")

    print("Conversion complete.")
    
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python conv_adj_to_metis.py <input.adj> <output.metis>")
        sys.exit(1)
    
    adj_file = sys.argv[1]
    metis_file = sys.argv[2]
    
    convert_adj_to_metis(adj_file, metis_file)