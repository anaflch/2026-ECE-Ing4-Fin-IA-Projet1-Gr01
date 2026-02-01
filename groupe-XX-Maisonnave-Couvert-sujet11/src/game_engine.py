import random

class Minesweeper:
    def __init__(self, width=10, height=10, num_mines=10):
        self.width = width
        self.height = height
        self.num_mines = num_mines
        self.grid = set()  # Set of (x,y) positions of mines
        self.revealed = set() # Set of revealed cells
        self.flags = set() # Set of flagged cells
        self.first_click = True # <--- NOUVEAU : On retient si c'est le début

    def _place_mines(self, safe_x, safe_y):
        """Place les mines aléatoirement MAIS évite la première case cliquée"""
        candidates = []
        for x in range(self.width):
            for y in range(self.height):
                # On ne met pas de mine sur le clic initial ni ses voisins immédiats
                # pour garantir une ouverture ("opening") sympa
                if abs(x - safe_x) <= 1 and abs(y - safe_y) <= 1:
                    continue
                candidates.append((x, y))
        
        # On choisit les mines parmi les candidats sûrs
        self.grid = set(random.sample(candidates, min(self.num_mines, len(candidates))))

    def reveal(self, x, y):
        """Révèle une case. Retourne True si c'est une mine (Perdu), False sinon."""
        if (x, y) in self.flags or (x, y) in self.revealed:
            return False
        
        # --- NOUVEAU : Génération des mines au premier clic ---
        if self.first_click:
            self._place_mines(x, y)
            self.first_click = False

        self.revealed.add((x, y))

        if (x, y) in self.grid:
            return True # BOOM

        # Si la case est vide (0 mine autour), on révèle les voisins (Flood Fill)
        if self.get_value(x, y) == 0:
            for nx, ny in self.get_neighbors(x, y):
                if (nx, ny) not in self.revealed:
                    self.reveal(nx, ny)
        
        return False

    def get_neighbors(self, x, y):
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0: continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    neighbors.append((nx, ny))
        return neighbors

    def get_value(self, x, y):
        """Retourne le nombre de mines autour de (x, y)"""
        if (x,y) in self.grid:
            return -1
        count = 0
        for nx, ny in self.get_neighbors(x, y):
            if (nx, ny) in self.grid:
                count += 1
        return count