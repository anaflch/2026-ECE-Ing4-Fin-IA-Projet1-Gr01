from __future__ import annotations

from typing import Dict, Hashable, Optional, Tuple
from pathlib import Path

import networkx as nx

Node = Hashable

# --------------------------------------------------------------------
# Palette de couleurs utilisée pour afficher les graphes colorés
# Les couleurs sont réutilisées si le nombre de couleurs dépasse la palette
# --------------------------------------------------------------------
def _palette():
    return [
        "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
        "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC", "#8CD17D",
    ]


def _safe_import_pyplot(show: bool):
    """
    Évite les crashs de backend quand on ne veut PAS afficher de fenêtre.
    - show=False => backend Agg
    - show=True  => backend par défaut (fenêtre)
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    return plt

# --------------------------------------------------------------------
# Crée le dossier parent du fichier de sortie si nécessaire
# --------------------------------------------------------------------
def _ensure_parent(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None
    p = Path(path)
    if p.parent and str(p.parent) != ".":
        p.parent.mkdir(parents=True, exist_ok=True)
    return p

# --------------------------------------------------------------------
# Affiche un graphe sans coloration (état initial)
# Utilisé pour montrer le graphe "avant" la coloration
# --------------------------------------------------------------------
def draw_plain(
    G: nx.Graph,
    pos: Optional[Dict[Node, Tuple[float, float]]] = None,
    title: str = "",
    save_path: Optional[str] = None,
    show: bool = False,
):
    plt = _safe_import_pyplot(show)
    save_p = _ensure_parent(save_path)

    if pos is None:
        pos = nx.spring_layout(G, seed=1)

    plt.figure(figsize=(7, 5))
     # Dessin du graphe sans couleur (nœuds blancs)
    nx.draw_networkx(
        G,
        pos=pos,
        with_labels=True,
        node_color="white",
        edgecolors="black",
        font_size=10,
    )
    if title:
        plt.title(title)
    plt.axis("off")

    if save_p:
        plt.savefig(save_p, bbox_inches="tight", dpi=200)
    if show:
        plt.show()
    plt.close()

# --------------------------------------------------------------------
# Affiche un graphe avec une coloration donnée
# Chaque couleur correspond à une valeur entière dans le dictionnaire coloring
# --------------------------------------------------------------------
def draw_coloring(
    G: nx.Graph,
    coloring: Dict[Node, int],
    pos: Optional[Dict[Node, Tuple[float, float]]] = None,
    title: str = "",
    save_path: Optional[str] = None,
    show: bool = False,
):
    plt = _safe_import_pyplot(show)
    save_p = _ensure_parent(save_path)

    pal = _palette()
    # Attribution d’une couleur à chaque nœud
    # Le modulo permet de gérer plus de couleurs que la taille de la palette
    node_colors = [pal[coloring.get(v, 0) % len(pal)] for v in G.nodes()]

    if pos is None:
        pos = nx.spring_layout(G, seed=1)

    plt.figure(figsize=(7, 5))
    nx.draw_networkx(
        G,
        pos=pos,
        with_labels=True,
        node_color=node_colors,
        edgecolors="black",
        font_size=10,
    )
    if title:
        plt.title(title)
    plt.axis("off")

    if save_p:
        plt.savefig(save_p, bbox_inches="tight", dpi=200)
    if show:
        plt.show()
    plt.close()
