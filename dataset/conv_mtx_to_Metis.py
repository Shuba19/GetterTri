#!/usr/bin/env python3
"""
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
"""

import argparse
import mmap
import io
from array import array

# Write buffer: 64 MB
WRITE_BUFFER = 64 * 1024 * 1024


def parse_args():
    parser = argparse.ArgumentParser(description="Convert MTX graph to METIS format")
    parser.add_argument("input", help="Input Matrix Market (.mtx) file")
    parser.add_argument("output", help="Output METIS file")
    parser.add_argument(
        "--zero-based-indices",
        action="store_true",
        help="Treat input matrix indices as 0-based (non-standard MTX)",
    )
    parser.add_argument(
        "--directed",
        action="store_true",
        help="Treat graph as directed (default: undirected)",
    )
    parser.add_argument(
        "--no-self-loops",
        action="store_true",
        help="Remove self-loops from the graph",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# MTX header parsing
# ---------------------------------------------------------------------------

def _read_header(mm):
    """
    Parse header + size line from an already-open mmap.
    Returns (data_start_pos, num_v, declared_entries, symmetry, comments).
    """
    first_line = mm.readline().decode()
    header = first_line.strip().split()
    if len(header) < 5 or header[0].lower() != "%%matrixmarket":
        raise ValueError("Invalid MTX header")
    if header[1].lower() != "matrix" or header[2].lower() != "coordinate":
        raise ValueError("Only 'matrix coordinate' MTX format is supported")
    symmetry = header[4].lower()

    comments = []
    while True:
        line = mm.readline()
        if not line:
            raise ValueError("MTX size line not found")
        s = line.lstrip()
        if s.startswith(b"%"):
            comments.append(s.decode().strip())
            continue
        parts = s.split(None, 3)
        if len(parts) < 3:
            raise ValueError("Invalid MTX size line")
        r, c, num_entries = int(parts[0]), int(parts[1]), int(parts[2])
        if r != c:
            raise ValueError(
                f"Matrix must be square for graph conversion ({r}x{c})"
            )
        return mm.tell(), r, num_entries, symmetry, comments


# ---------------------------------------------------------------------------
# Compact adjacency: list of array('l') — much less RAM than list-of-set
# ---------------------------------------------------------------------------

def _build_adj(mm, data_pos, num_v, directed, remove_self_loops, zero_based_indices):
    """
    Single pass over edge data: builds adjacency as list of array('l').
    Deduplication via sort+unique after loading.
    """
    offset = 1 if zero_based_indices else 0
    adj = [array("l") for _ in range(num_v)]

    mm.seek(data_pos)
    while True:
        line = mm.readline()
        if not line:
            break
        if line[0:1] in (b"%", b"\n", b"\r"):
            continue

        parts = line.split(None, 2)
        if len(parts) < 2:
            continue

        u = int(parts[0]) - 1 + offset   # 0-based internal
        v = int(parts[1]) - 1 + offset

        if not (0 <= u < num_v and 0 <= v < num_v):
            raise ValueError(f"Vertex id out of range [1,{num_v}]: ({u+1},{v+1})")

        if remove_self_loops and u == v:
            continue

        adj[u].append(v)
        if not directed and u != v:
            adj[v].append(u)

    # Sort + deduplicate each neighbour list in-place
    for i in range(num_v):
        if len(adj[i]) > 1:
            adj[i] = array("l", sorted(set(adj[i])))

    return adj


# ---------------------------------------------------------------------------
# Streaming write: never builds the full output in memory
# ---------------------------------------------------------------------------

def _write_metis_streaming(filepath, adj, num_v, directed):
    """
    Count edges (O(V) pass over lengths), then write with a large
    BufferedWriter to minimise syscalls without holding all text in RAM.
    """
    total_half = sum(len(a) for a in adj)
    num_edges = total_half if directed else total_half // 2

    with io.open(filepath, "w", buffering=WRITE_BUFFER, encoding="ascii") as f:
        f.write(f"{num_v} {num_edges}\n")
        for nbrs in adj:
            if nbrs:
                f.write(" ".join(str(v + 1) for v in nbrs))
            f.write("\n")

    return num_edges


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print(f"[*] Reading MTX file: {args.input}")

    with open(args.input, "rb") as raw:
        mm = mmap.mmap(raw.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            data_pos, num_v, declared_entries, symmetry, comments = _read_header(mm)

            print(f"    Declared entries : {declared_entries}")
            print(f"    Matrix symmetry  : {symmetry}")
            print(f"    Unique vertices  : {num_v}")
            if comments:
                print(f"    Comments found   : {len(comments)}")
                for c in comments[:3]:
                    print(f"      {c}")
                if len(comments) > 3:
                    print(f"      ... ({len(comments) - 3} more)")

            print("[*] Building adjacency list …")
            adj = _build_adj(
                mm, data_pos, num_v,
                directed=args.directed,
                remove_self_loops=args.no_self_loops,
                zero_based_indices=args.zero_based_indices,
            )
        finally:
            mm.close()

    raw_edges = sum(len(a) for a in adj)
    if not args.directed:
        raw_edges //= 2
    print(f"    Raw edges (after dedup) : {raw_edges}")

    print(f"[*] Writing METIS file: {args.output}")
    num_edges = _write_metis_streaming(args.output, adj, num_v, directed=args.directed)
    print(f"    Vertices written : {num_v}")
    print(f"    Edges written    : {num_edges}")
    print("[+] Done.")


if __name__ == "__main__":
    main()