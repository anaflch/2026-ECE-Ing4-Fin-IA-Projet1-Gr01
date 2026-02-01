from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional, Tuple

import networkx as nx

from instances import load_instance
from heuristics import greedy_coloring, dsatur_coloring
from solve_coloring import solve_min_coloring

Node = Hashable

# --------------------------------------------------------------------
# Structure de données pour stocker une ligne de benchmark
# Chaque instance testée + méthode correspond à une ligne du CSV final
# --------------------------------------------------------------------
@dataclass
class BenchRow:
    instance: str
    family: str
    params: str
    seed: int
    method: str
    colors_used: int
    valid: bool                 # la coloration est-elle valide ?
    time_s: float
    status: str
    k_found: Optional[int]

# --------------------------------------------------------------------
# Vérifie qu’une coloration est valide :
# deux sommets adjacents ne doivent pas avoir la même couleur
# --------------------------------------------------------------------
def is_valid_coloring(G: nx.Graph, coloring: Dict[Node, int]) -> bool:
    for u, v in G.edges():
        if coloring.get(u) == coloring.get(v):
            return False
    return True

# --------------------------------------------------------------------
# Compte le nombre de couleurs distinctes utilisées dans la coloration
# --------------------------------------------------------------------
def colors_used(coloring: Dict[Node, int]) -> int:
    return len(set(coloring.values())) if coloring else 0

# --------------------------------------------------------------------
# Crée automatiquement le dossier parent du fichier de sortie (CSV)
# --------------------------------------------------------------------
def ensure_parent_dir(path: str) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

# --------------------------------------------------------------------
# Lance une campagne complète de benchmarks sur différentes instances
# et différentes méthodes, puis écrit les résultats dans un CSV
# --------------------------------------------------------------------
def run_benchmark(
    out_csv: str = "outputs/benchmark.csv",
    seeds: List[int] = [1, 2, 3],
    methods: List[str] = ["greedy", "dsatur", "cp_min"],

    # Paramètres des graphes testés
    erdos_sizes: List[int] = [30, 50, 80],
    erdos_ps: List[float] = [0.10, 0.20, 0.30],
    grids: List[Tuple[int, int]] = [(8, 8), (12, 12)],
    
    include_map_like: bool = True,
    timeout_cp_min: float = 2.0,
    kmax: Optional[int] = None,
) -> List[BenchRow]:
    # Liste qui contiendra toutes les lignes du benchmark
    rows: List[BenchRow] = []

    # 1) Instance map_like
    if include_map_like:
        inst = load_instance("map_like")
        G = inst.graph
        for method in methods:
            t0 = time.perf_counter()
            status = "OK"
            k_found = None

            # Sélection de la méthode
            if method == "greedy":
                coloring = greedy_coloring(G)
            elif method == "dsatur":
                coloring = dsatur_coloring(G)
            elif method == "cp_min":
                best_k, coloring, log = solve_min_coloring(
                    nodes=list(G.nodes()),
                    edges=list(G.edges()),
                    k_max=kmax,
                    timeout_per_k_s=timeout_cp_min,
                )
                k_found = best_k
                status = "FOUND" if coloring is not None else "NOT_FOUND"
            else:
                continue

            dt = time.perf_counter() - t0
            valid = coloring is not None and is_valid_coloring(G, coloring)
            used = colors_used(coloring) if coloring is not None else 0

            # Ajout d’une ligne de benchmark
            rows.append(BenchRow(
                instance=inst.name,
                family="map_like",
                params="",
                seed=0,
                method=method,
                colors_used=used,
                valid=valid,
                time_s=dt,
                status=status,
                k_found=k_found,
            ))

     # 2) Grilles (grid)
    for (w, h) in grids:
        inst = load_instance("grid", w=w, h=h)
        G = inst.graph
        for method in methods:
            t0 = time.perf_counter()
            status = "OK"
            k_found = None

            if method == "greedy":
                coloring = greedy_coloring(G)
            elif method == "dsatur":
                coloring = dsatur_coloring(G)
            elif method == "cp_min":
                best_k, coloring, log = solve_min_coloring(
                    nodes=list(G.nodes()),
                    edges=list(G.edges()),
                    k_max=kmax,
                    timeout_per_k_s=timeout_cp_min,
                )
                k_found = best_k
                status = "FOUND" if coloring is not None else "NOT_FOUND"
            else:
                continue

            dt = time.perf_counter() - t0
            valid = coloring is not None and is_valid_coloring(G, coloring)
            used = colors_used(coloring) if coloring is not None else 0

            rows.append(BenchRow(
                instance=inst.name,
                family="grid",
                params=f"w={w};h={h}",
                seed=0,
                method=method,
                colors_used=used,
                valid=valid,
                time_s=dt,
                status=status,
                k_found=k_found,
            ))

    # 3) Graphes d'Erdos
    for n in erdos_sizes:
        for p in erdos_ps:
            for seed in seeds:
                inst = load_instance("erdos", n=n, p=p, seed=seed)
                G = inst.graph

                for method in methods:
                    t0 = time.perf_counter()
                    status = "OK"
                    k_found = None

                    if method == "greedy":
                        coloring = greedy_coloring(G)
                    elif method == "dsatur":
                        coloring = dsatur_coloring(G)
                    elif method == "cp_min":
                        best_k, coloring, log = solve_min_coloring(
                            nodes=list(G.nodes()),
                            edges=list(G.edges()),
                            k_max=kmax,
                            timeout_per_k_s=timeout_cp_min,
                        )
                        k_found = best_k
                        status = "FOUND" if coloring is not None else "NOT_FOUND"
                    else:
                        continue

                    dt = time.perf_counter() - t0
                    valid = coloring is not None and is_valid_coloring(G, coloring)
                    used = colors_used(coloring) if coloring is not None else 0

                    rows.append(BenchRow(
                        instance=inst.name,
                        family="erdos",
                        params=f"n={n};p={p}",
                        seed=seed,
                        method=method,
                        colors_used=used,
                        valid=valid,
                        time_s=dt,
                        status=status,
                        k_found=k_found,
                    ))

    # 4) Écriture du CSV
    ensure_parent_dir(out_csv)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)

        # En-tête
        w.writerow([
            "instance", "family", "params", "seed", "method",
            "colors_used", "valid", "time_s", "status", "k_found"
        ])
        # Lignes de résultats
        for r in rows:
            w.writerow([
                r.instance, r.family, r.params, r.seed, r.method,
                r.colors_used, int(r.valid), f"{r.time_s:.6f}", r.status,
                "" if r.k_found is None else r.k_found
            ])

    return rows
