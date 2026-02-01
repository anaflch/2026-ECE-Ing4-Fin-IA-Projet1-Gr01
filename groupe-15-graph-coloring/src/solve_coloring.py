from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional, Tuple

from ortools.sat.python import cp_model

Node = Hashable
Edge = Tuple[Node, Node]

# --------------------------------------------------------------------
# Structure contenant les informations retournées par le solveur CP-SAT
# --------------------------------------------------------------------
@dataclass(frozen=True)
class SolveInfo:
    status: str
    time_s: float
    conflicts: int
    branches: int

# --------------------------------------------------------------------
# Conversion du code de statut OR-Tools vers une chaîne lisible
# --------------------------------------------------------------------
def _status(code: int) -> str:
    return {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
        cp_model.UNKNOWN: "UNKNOWN",
    }.get(code, "UNKNOWN")

# --------------------------------------------------------------------
# Génère une solution gloutonne utilisée comme "hint" pour CP-SAT
# Cela aide souvent le solveur à converger plus rapidement
# --------------------------------------------------------------------
def _greedy_hint(nodes: List[Node], edges: List[Edge]) -> Dict[Node, int]:
    # Construction de la liste d’adjacence et du degré de chaque nœud
    adj = {v: set() for v in nodes}
    deg = {v: 0 for v in nodes}
    for u, v in edges:
        if u != v and u in adj and v in adj:
            adj[u].add(v); adj[v].add(u)
            deg[u] += 1; deg[v] += 1

    order = sorted(nodes, key=lambda x: deg[x], reverse=True) # Ordre décroissant des nœuds selon leur degré
    col: Dict[Node, int] = {}
    for v in order:
        used = {col[u] for u in adj[v] if u in col}
        c = 0
        while c in used:
            c += 1
        col[v] = c
    return col

# --------------------------------------------------------------------
# Résolution du problème de k-coloration :
# -chaque noeud a une couleur entre 0 et k-1
# -2 nœuds adjacents ne peuvent pas avoir la même couleur
# --------------------------------------------------------------------
def solve_k_coloring(
    nodes: List[Node],
    edges: List[Edge],
    k: int,
    timeout_s: float = 5.0,
    num_workers: int = 8,
    symmetry_breaking: bool = True,
    use_hints: bool = True,
) -> Tuple[Optional[Dict[Node, int]], SolveInfo]:
    #Résout un problème de k-coloration avec CP-SAT.
    if k < 1:
        raise ValueError("k must be >= 1")

    nodes = list(nodes)
    if not nodes:
        return {}, SolveInfo("OPTIMAL", 0.0, 0, 0)

    edges = list(edges)
    # Création du modèle CP-SAT
    model = cp_model.CpModel()
    c = {v: model.NewIntVar(0, k - 1, f"c_{v}") for v in nodes}

     # on fixe arbitrairement la couleur du premier nœud à 0
    if symmetry_breaking:
        model.Add(c[nodes[0]] == 0)

    # Contraintes: 2 sommets adjacents doivent avoir des couleurs différentes
    for u, v in edges:
        if u != v and u in c and v in c:
            model.Add(c[u] != c[v])

    # Ajout de hints (solution initiale) via une heuristique gloutonne(greedy)
    if use_hints:
        hint = _greedy_hint(nodes, edges)
        if max(hint.values(), default=-1) < k:
            for v, hv in hint.items():
                model.AddHint(c[v], int(hv))

    # Paramétrage du solveur
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(timeout_s)
    solver.parameters.num_search_workers = int(num_workers)

    # Lancement de la résolution
    st = solver.Solve(model)
    # Récupération des statistiques de résolution
    info = SolveInfo(
        status=_status(st),
        time_s=float(solver.WallTime()),
        conflicts=int(solver.NumConflicts()),
        branches=int(solver.NumBranches()),
    )
    # Si une solution est trouvée: on la retourne
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {v: int(solver.Value(c[v])) for v in nodes}, info
    return None, info

# --------------------------------------------------------------------
# Recherche de la coloration minimale :
# on teste successivement k=k_min, k_min+1,k_max
# --------------------------------------------------------------------
def solve_min_coloring(
    nodes: List[Node],
    edges: List[Edge],
    # bornes
    k_min: int = 1,
    k_max: Optional[int] = None,
    # paramètres du solveur
    timeout_per_k_s: float = 3.0,
    num_workers: int = 8,
    symmetry_breaking: bool = True,
) -> Tuple[Optional[int], Optional[Dict[Node, int]], List[Tuple[int, SolveInfo]]]:
    # Recherche la coloration minimale en augmentant progressivement k.
    nodes = list(nodes)
    if not nodes:
        return 0, {}, []
    # Par défaut, on autorise jusqu’à n couleurs
    if k_max is None:
        k_max = len(nodes)

    k_min = max(1, int(k_min))
    k_max = min(int(k_max), len(nodes))
    if k_min > k_max:
        return None, None, []

    log: List[Tuple[int, SolveInfo]] = []
    for k in range(k_min, k_max + 1):
        sol, info = solve_k_coloring(
            nodes=nodes,
            edges=edges,
            k=k,
            timeout_s=timeout_per_k_s,
            num_workers=num_workers,
            symmetry_breaking=symmetry_breaking,
            use_hints=True,
        )
        log.append((k, info))
        # Dès qu’une solution existe, k est minimal
        if sol is not None:
            return k, sol, log

    # Aucune solution trouvée dans les bornes
    return None, None, log
