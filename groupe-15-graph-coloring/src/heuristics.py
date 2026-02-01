from __future__ import annotations

from typing import Dict, Hashable, Optional, List
import networkx as nx

Node = Hashable


def greedy_coloring(G: nx.Graph, order: Optional[List[Node]] = None) -> Dict[Node, int]:
    """
    Coloriage glouton: assigne au sommet la plus petite couleur disponible.
    """
    if order is None:
        order = list(G.nodes())

    coloring: Dict[Node, int] = {}
    for v in order:
        used = {coloring[u] for u in G.neighbors(v) if u in coloring}
        c = 0
        while c in used:
            c += 1
        coloring[v] = c
    return coloring


def dsatur_coloring(G: nx.Graph) -> Dict[Node, int]:
    """
    DSATUR: à chaque étape choisit le sommet au plus grand degré de saturation,
    puis tie-break avec le degré.
    """
    nodes = list(G.nodes())
    if not nodes:
        return {}

    coloring: Dict[Node, int] = {}
    sat_colors: Dict[Node, set[int]] = {v: set() for v in nodes}
    degree = {v: G.degree(v) for v in nodes}
    uncolored = set(nodes)

    while uncolored:
        v = max(uncolored, key=lambda x: (len(sat_colors[x]), degree[x]))
        used = {coloring[u] for u in G.neighbors(v) if u in coloring}
        c = 0
        while c in used:
            c += 1
        coloring[v] = c
        uncolored.remove(v)
        for u in G.neighbors(v):
            if u in uncolored:
                sat_colors[u].add(c)

    return coloring
