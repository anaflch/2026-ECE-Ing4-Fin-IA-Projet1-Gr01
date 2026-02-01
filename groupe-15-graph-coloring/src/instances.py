from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, Optional, Tuple

import networkx as nx

Node = Hashable

# --------------------------------------------------------------------
# Structure représentant une instance de problème de coloration
# -name: nom de l’instance (utilisé pour l’affichage et les exports)
# -graph: graphe NetworkX
# -pos: positions des nœuds (optionnelles, pour la visualisation)
# --------------------------------------------------------------------
@dataclass(frozen=True)
class Instance:
    name: str
    graph: nx.Graph
    pos: Optional[Dict[Node, Tuple[float, float]]] = None

# --------------------------------------------------------------------
# Normalise le nom d’une instance pour le rendre robuste aux erreurs
# Exemples acceptés: "map.like", "map-like", "map like" au lieu de "map_like"
# --------------------------------------------------------------------
def _norm_name(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace(".", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )

# --------------------------------------------------------------------
# Instance simple: triangle (3 sommets)
# -nécessite 3 couleurs
# -utile pour tester un cas trivial
# --------------------------------------------------------------------
def triangle() -> Instance:
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (0, 2)])
    pos = {0: (0, 0), 1: (1, 0), 2: (0.5, 0.9)}
    return Instance("triangle", G, pos)

# --------------------------------------------------------------------
# Cycle de n sommets:
# -si n est pair: 2-coloriable
# -si n est impair: nécessite 3 couleurs
# --------------------------------------------------------------------
def cycle(n: int = 8) -> Instance:
    n = max(3, int(n)) # Un cycle doit avoir au moins 3 sommets
    G = nx.cycle_graph(n)
    pos = nx.circular_layout(G) # Disposition circulaire pour la visualisation
    return Instance(f"cycle_{n}", G, pos)

# --------------------------------------------------------------------
# Grille w×h (graphe planaire):
# -typiquement 2-coloriable (comme un damier)
# -utilisée pour tester des graphes structurés
# --------------------------------------------------------------------
def grid(w: int = 4, h: int = 4) -> Instance:
    w = max(2, int(w)) # Une grille doit avoir au moins 2 lignes et 2 colonnes
    h = max(2, int(h))
    G = nx.grid_2d_graph(w, h)
    pos = {(x, y): (float(x), float(-y)) for (x, y) in G.nodes()}  # Positions fixes (inversion de y pour un affichage naturel)
    return Instance(f"grid_{w}x{h}", G, pos)

# --------------------------------------------------------------------
# Graphe aléatoire d’Erdős–Rényi:
# -n: nombre de sommets
# -p: probabilité de connexion entre deux sommets
# -seed: graine pour la reproductibilité
# --------------------------------------------------------------------
def random_erdos(n: int = 25, p: float = 0.2, seed: int = 1) -> Instance:
    n = max(1, int(n))
    p = float(p)
    p = 0.0 if p < 0.0 else 1.0 if p > 1.0 else p # On force p à rester dans [0, 1]
    seed = int(seed)
    G = nx.erdos_renyi_graph(n=n, p=p, seed=seed)
    pos = nx.spring_layout(G, seed=seed)
    return Instance(f"erdos_n{n}_p{p}_s{seed}", G, pos)

# --------------------------------------------------------------------
# Carte fictive (régions A à J):
# -représente un problème de coloration de carte
# -graphe planaire
# -utilisé pour illustrer le théorème des 4 couleurs
# --------------------------------------------------------------------
def map_like() -> Instance:
    """Petite 'carte' fictive (régions A..J) pour démo de map coloring."""
    regions = list("ABCDEFGHIJ")
    # Liste des adjacences entre régions
    adj = [
        ("A", "B"), ("A", "C"),
        ("B", "C"), ("B", "D"),
        ("C", "D"), ("C", "E"),
        ("D", "E"), ("D", "F"),
        ("E", "F"), ("E", "G"),
        ("F", "G"), ("F", "H"),
        ("G", "H"), ("G", "I"),
        ("H", "I"), ("H", "J"),
        ("I", "J"),
    ]
    G = nx.Graph()
    G.add_nodes_from(regions)
    G.add_edges_from(adj)

    # Positions choisies manuellement pour ressembler à une carte
    pos = {
        "A": (0, 2), "B": (1, 2), "C": (0.5, 1.5),
        "D": (1.5, 1.5), "E": (1, 1),
        "F": (2, 1), "G": (1.5, 0.5),
        "H": (2.5, 0.5), "I": (2, 0),
        "J": (3, 0),
    }
    return Instance("map_like", G, pos)

# --------------------------------------------------------------------
# Fonction centrale de chargement des instances
# Sélectionne la bonne instance en fonction du nom fourni
# --------------------------------------------------------------------
def load_instance(
    name: str,
    n: int = 25,
    p: float = 0.2,
    seed: int = 1,
    w: int = 4,
    h: int = 4,
) -> Instance:
    key = _norm_name(name)  # Normalisation du nom pour éviter les erreurs utilisateur

    if key == "triangle":
        return triangle()
    if key == "cycle":
        return cycle(n=n)
    if key == "grid":
        return grid(w=w, h=h)
    if key in ("random", "erdos", "erdos_renyi"):
        return random_erdos(n=n, p=p, seed=seed)
    if key in ("map", "map_like"):
        return map_like()
    # Erreur claire si l’instance n’est pas reconnue
    raise ValueError(f"Instance inconnue: {name} (triangle/cycle/grid/erdos/map_like)")
