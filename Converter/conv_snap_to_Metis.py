#!/usr/bin/env python3
"""
SNAP to METIS graph format converter.

SNAP format:
  # comment lines
  src dst          (one edge per line, 0-based or 1-based node IDs)

METIS format:
  <num_vertices> <num_edges> [fmt] [ncon]
  <adj1> <adj2> ...   (one line per vertex, 1-based, undirected)
"""

import sys
import argparse
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Convert SNAP graph to METIS format")
    parser.add_argument("input",  help="Input SNAP file")
    parser.add_argument("output", help="Output METIS file")
    parser.add_argument(
        "--zero-based", action="store_true",
        help="Input node IDs are 0-based (default: auto-detect)"
    )
    parser.add_argument(
        "--one-based", action="store_true",
        help="Input node IDs are 1-based"
    )
    parser.add_argument(
        "--directed", action="store_true",
        help="Treat graph as directed (default: undirected — adds reverse edges)"
    )
    parser.add_argument(
        "--no-self-loops", action="store_true",
        help="Remove self-loops from the graph"
    )
    return parser.parse_args()


def detect_base(edges):
    """Auto-detect if node IDs are 0-based or 1-based."""
    all_nodes = set()
    for u, v in edges:
        all_nodes.add(u)
        all_nodes.add(v)
    return 0 if 0 in all_nodes else 1


def read_snap(filepath, remove_self_loops=False):
    """Read SNAP file, return raw edge list and comment lines."""
    edges = []
    comments = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#") or line.startswith("%"):
                comments.append(line)
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            if remove_self_loops and u == v:
                continue
            edges.append((u, v))

    return edges, comments


def build_adjacency(edges, base, directed=False):
    """
    Build adjacency list (1-based for METIS).
    Remaps node IDs to a contiguous 1-based range.
    """
    # Collect all node IDs and remap to contiguous 1-based
    all_nodes = sorted(set(n for e in edges for n in e))
    remap = {old: (new + 1) for new, old in enumerate(all_nodes)}
    num_v = len(all_nodes)

    adj = defaultdict(set)

    for u, v in edges:
        u_new = remap[u]
        v_new = remap[v]
        if directed:
            adj[u_new].add(v_new)
        else:
            # Undirected: add both directions
            adj[u_new].add(v_new)
            adj[v_new].add(u_new)

    # Ensure every vertex has an entry (even if isolated)
    for node in remap.values():
        if node not in adj:
            adj[node] = set()

    return adj, num_v, remap


def write_metis(filepath, adj, num_v, directed=False):
    """Write adjacency list to METIS format."""
    # METIS counts undirected edges (each edge once)
    total_edges = sum(len(nbrs) for nbrs in adj.values())
    if not directed:
        # Each undirected edge appears in two adjacency lists
        total_edges //= 2

    with open(filepath, "w") as f:
        # Header: num_vertices num_edges
        f.write(f"{num_v} {total_edges}\n")

        for node in range(1, num_v + 1):
            neighbors = sorted(adj.get(node, set()))
            f.write(" ".join(map(str, neighbors)) + "\n")

    return total_edges


def main():
    args = parse_args()

    print(f"[*] Reading SNAP file: {args.input}")
    edges, comments = read_snap(args.input, remove_self_loops=args.no_self_loops)
    print(f"    Raw edges read   : {len(edges)}")
    if comments:
        print(f"    Comments found   : {len(comments)}")
        for c in comments[:3]:
            print(f"      {c}")
        if len(comments) > 3:
            print(f"      ... ({len(comments) - 3} more)")

    # Determine base
    if args.one_based:
        base = 1
    elif args.zero_based:
        base = 0
    else:
        base = detect_base(edges)
        print(f"    Auto-detected base: {base}-based node IDs")

    # Build adjacency
    adj, num_v, remap = build_adjacency(edges, base, directed=args.directed)
    print(f"    Unique vertices  : {num_v}")

    # Write METIS
    print(f"[*] Writing METIS file: {args.output}")
    num_edges = write_metis(args.output, adj, num_v, directed=args.directed)
    print(f"    Vertices written : {num_v}")
    print(f"    Edges written    : {num_edges}")
    print("[+] Done.")


if __name__ == "__main__":
    main()