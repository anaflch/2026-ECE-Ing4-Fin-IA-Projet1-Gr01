# --- IMPORTATIONS ---
from dataclasses import dataclass
from typing import List, Tuple
import random

# --- STRUCTURE DE DONNÉES ---
# Cette classe stocke les infos d'un emplacement de mot (Slot)
@dataclass
class WordSlot:
    id: int             # Identifiant unique (0, 1, 2...)
    direction: str      # 'H' (Horizontal) ou 'V' (Vertical)
    row: int            # Ligne de départ
    col: int            # Colonne de départ
    length: int         # Longueur du mot
    cells: List[Tuple[int, int]] # Liste des coordonnées (row, col) occupées

# --- ANALYSEUR DE GRILLE ---
class GridStructure:
    def __init__(self, grid_layout: List[str]):
        """
        grid_layout: Liste de strings représentant la grille.
                     '#' = case noire, '.' = case blanche
        """
        self.grid = grid_layout
        self.rows = len(grid_layout)
        self.cols = len(grid_layout[0])
        
        self.slots = []           # Liste de tous les WordSlot
        self.intersections = []   # Liste des croisements
        
        # On lance l'analyse tout de suite
        self._parse_slots()
        self._find_intersections()

    def _parse_slots(self):
        """Détecte les mots horizontaux et verticaux."""
        slot_id = 0
        self.slots = []

        # 1. Analyse Horizontale
        for r in range(self.rows):
            current_cells = []
            for c in range(self.cols):
                if self.grid[r][c] == '.':
                    current_cells.append((r, c))
                else:
                    # Case noire : fin de mot
                    if len(current_cells) >= 1:
                        self.slots.append(WordSlot(slot_id, 'H', current_cells[0][0], current_cells[0][1], len(current_cells), current_cells))
                        slot_id += 1
                    current_cells = []
            # Fin de ligne
            if len(current_cells) >= 1:
                self.slots.append(WordSlot(slot_id, 'H', current_cells[0][0], current_cells[0][1], len(current_cells), current_cells))
                slot_id += 1

        # 2. Analyse Verticale
        for c in range(self.cols):
            current_cells = []
            for r in range(self.rows):
                if self.grid[r][c] == '.':
                    current_cells.append((r, c))
                else:
                    # Case noire : fin de mot
                    if len(current_cells) >= 1:
                        self.slots.append(WordSlot(slot_id, 'V', current_cells[0][0], current_cells[0][1], len(current_cells), current_cells))
                        slot_id += 1
                    current_cells = []
            # Fin de colonne
            if len(current_cells) >= 1:
                self.slots.append(WordSlot(slot_id, 'V', current_cells[0][0], current_cells[0][1], len(current_cells), current_cells))
                slot_id += 1

    def _find_intersections(self):
        """Trouve où les mots se croisent."""
        # On compare chaque slot Horizontal avec chaque slot Vertical
        horiz_slots = [s for s in self.slots if s.direction == 'H']
        vert_slots = [s for s in self.slots if s.direction == 'V']

        for h_slot in horiz_slots:
            for v_slot in vert_slots:
                # Vérifie s'ils partagent une cellule commune
                # (Intersection des ensembles de coordonnées)
                common_cells = set(h_slot.cells).intersection(set(v_slot.cells))
                
                if common_cells:
                    # Il y a croisement ! (Normalement un seul point)
                    coord = list(common_cells)[0]
                    
                    # À quel index (0, 1, 2...) de chaque mot se trouve le croisement ?
                    h_index = h_slot.cells.index(coord)
                    v_index = v_slot.cells.index(coord)
                    
                    self.intersections.append({
                        'id_h': h_slot.id,      # ID du mot horizontal
                        'id_v': v_slot.id,      # ID du mot vertical
                        'index_h': h_index,     # Index de la lettre dans le mot H
                        'index_v': v_index      # Index de la lettre dans le mot V
                    })

    def print_report(self):
        """Affiche un résumé de l'analyse dans la console."""
        print(f"\n--- Analyse de la Grille {self.rows}x{self.cols} ---")
        print(f"Nombre de mots à trouver (Slots) : {len(self.slots)}")
        print(f"Nombre de croisements (Contraintes) : {len(self.intersections)}")
        
        print("\n--- Liste des Slots ---")
        for s in self.slots:
            print(f"ID {s.id} ({s.direction}) : Pos({s.row},{s.col}), Longueur {s.length}")
            
        print("\n--- Exemples de Croisements ---")
        for i, inter in enumerate(self.intersections[:5]): # Affiche les 5 premiers
            print(f"Le mot ID {inter['id_h']} (idx {inter['index_h']}) croise le mot ID {inter['id_v']} (idx {inter['index_v']})")

# --- TEST UNITAIRE ---
if __name__ == "__main__":
    # --- GÉNÉRATION D'UNE GRILLE ALÉATOIRE 5x5 ---
    ROWS, COLS = 12, 12
    NB_NOIRES = 20

    # 1. Création grille vide
    # On utilise une liste de listes pour pouvoir modifier facilement
    temp_grid = [['.' for _ in range(COLS)] for _ in range(ROWS)]

    # 2. Ajout des cases noires aléatoires
    count = 0
    attempts = 0
    while count < NB_NOIRES and attempts < 200:
        attempts += 1
        r = random.randint(0, ROWS - 1)
        c = random.randint(0, COLS - 1)
        if temp_grid[r][c] == '#': continue
        
        # Pas de contact
        if any(0 <= r+dr < ROWS and 0 <= c+dc < COLS and temp_grid[r+dr][c+dc] == '#' for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]):
            continue
        
        # Max 3 sur bords
        if r in [0, ROWS-1] or c in [0, COLS-1]:
            if sum(1 for i in range(ROWS) for j in range(COLS) if temp_grid[i][j] == '#' and (i in [0, ROWS-1] or j in [0, COLS-1])) >= 3:
                continue

        temp_grid[r][c] = '#'
        count += 1

    # Conversion en format attendu par GridStructure (liste de strings)
    grille_test = ["".join(row) for row in temp_grid]

    # 3. Affichage visuel (A-L, 1-12)
    print(f"\n--- Grille Générée ({ROWS}x{COLS} avec {NB_NOIRES} cases noires) ---")
    print("   " + " ".join([chr(65 + i) for i in range(COLS)])) # A B C ...
    for i, row in enumerate(grille_test):
        print(f"{str(i + 1).rjust(2)} {' '.join(list(row))}")

    print("\n--- Lancement de l'analyse ---")
    # On crée l'analyseur avec notre grille test
    analyseur = GridStructure(grille_test)
    # On affiche le résultat
    analyseur.print_report()