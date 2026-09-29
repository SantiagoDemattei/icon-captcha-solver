import hashlib
from dataclasses import dataclass


@dataclass
class PackedGraph:
    nodes: int
    offsets: list[int]
    neighbors: list[int]
    edge_indices: list[int]


def _seed_bytes(seed) -> bytes:
    if seed is None or seed == "":
        return b""
    return str(seed).encode("utf-8")


def edge_endpoints(edge: int, num_nodes: int, seed: bytes) -> tuple[int, int]:
    h = hashlib.sha256(edge.to_bytes(4, "little") + seed).digest()
    u = int.from_bytes(h[0:8], "little") % num_nodes
    v = int.from_bytes(h[8:16], "little") % num_nodes
    return u, v


def build_graph(params: dict) -> PackedGraph:
    num_nodes = 1 << params["NODES_BITS"]
    num_edges = 1 << params["EDGES_BITS"]
    seed = _seed_bytes(params.get("GRAPH_SEED"))

    us = [0] * num_edges
    vs = [0] * num_edges
    degree = [0] * num_nodes
    for e in range(num_edges):
        u, v = edge_endpoints(e, num_nodes, seed)
        us[e] = u
        vs[e] = v
        degree[u] += 1
        degree[v] += 1

    offsets = [0] * (num_nodes + 1)
    for i in range(num_nodes):
        offsets[i + 1] = offsets[i] + degree[i]

    neighbors = [0] * (2 * num_edges)
    edge_indices = [0] * (2 * num_edges)
    cursor = offsets[:num_nodes]
    for e in range(num_edges):
        u, v = us[e], vs[e]
        p = cursor[u]
        cursor[u] = p + 1
        neighbors[p] = v
        edge_indices[p] = e
        p = cursor[v]
        cursor[v] = p + 1
        neighbors[p] = u
        edge_indices[p] = e

    return PackedGraph(num_nodes, offsets, neighbors, edge_indices)


def compute_start_index(challenge_hex: str, num_nodes: int) -> int:
    h = hashlib.sha256(bytes.fromhex(challenge_hex)).digest()
    return int.from_bytes(h[0:8], "little") % num_nodes


def find_cycle(g: PackedGraph, cycle_length: int, start_offset: int) -> list[int] | None:
    offsets, neighbors, edge_indices = g.offsets, g.neighbors, g.edge_indices
    nodes = g.nodes
    on_path = bytearray(nodes)

    def dfs(start: int) -> list[int] | None:
        on_path[start] = 1
        edge_path: list[int] = []
        stack = [[start, -1, -1, 0, offsets[start], offsets[start + 1]]]
        while stack:
            top = stack[-1]
            it_pos, it_end = top[4], top[5]
            if it_pos >= it_end:
                on_path[top[0]] = 0
                if top[2] != -1:
                    edge_path.pop()
                stack.pop()
                continue
            top[4] = it_pos + 1
            nb = neighbors[it_pos]
            eidx = edge_indices[it_pos]
            if nb == top[1]:
                continue
            depth = top[3]
            if on_path[nb]:
                # el servidor recalcula el mismo recorrido determinista para verificar --
                # tiene que ser el primer ciclo encontrado, no cualquier ciclo valido.
                if nb == start and depth + 1 == cycle_length:
                    edge_path.append(eidx)
                    return edge_path[:]
            elif depth + 1 < cycle_length:
                on_path[nb] = 1
                edge_path.append(eidx)
                stack.append([nb, top[0], eidx, depth + 1, offsets[nb], offsets[nb + 1]])
        return None

    for k in range(nodes):
        found = dfs((start_offset + k) % nodes)
        if found is not None:
            return found
    return None


def solve_challenges(challenges: list[str], params: dict, graph: PackedGraph | None = None) -> list[dict]:
    g = graph or build_graph(params)
    cycle_length = params["CYCLE_LENGTH"]
    out = []
    for ch in challenges:
        cycle = find_cycle(g, cycle_length, compute_start_index(ch, g.nodes))
        if cycle is None:
            raise RuntimeError(f"No se encontro un ciclo de longitud {cycle_length} para el challenge {ch}")
        out.append({"challenge": ch, "edgeIndices": cycle})
    return out
