import random

class CSPSolver:
    def __init__(self, game, verbose=True): # Ajout du paramètre verbose
        self.game = game
        self.MAX_BACKTRACK_VARS = 14
        self.verbose = verbose # On stocke l'info

    def solve(self):
        moves = set()
        flags = set()
        
        # --- 1. LOGIQUE SIMPLE (Rapide) ---
        simple_found = False
        for (x, y) in list(self.game.revealed):
            val = self.game.get_value(x, y)
            if val == 0: continue 

            neighbors = self.game.get_neighbors(x, y)
            hidden = [n for n in neighbors if n not in self.game.revealed and n not in self.game.flags]
            flagged = [n for n in neighbors if n in self.game.flags]
            
            if not hidden: continue

            # Si le nombre de drapeaux = le chiffre, le reste est sûr
            if len(flagged) == val:
                for n in hidden:
                    moves.add(n)
                    simple_found = True

            # Si le nombre de cases cachées + drapeaux = le chiffre, tout est mine
            elif len(hidden) + len(flagged) == val:
                for n in hidden:
                    flags.add(n)
                    simple_found = True

        if simple_found:
            # On ne print pas ici pour ne pas spammer la console quand c'est facile
            return list(moves), list(flags)

        # --- 2. BACKTRACKING INTELLIGENT (Expert) ---
        if self.verbose: print("🔍 Logique simple épuisée. Tentative de Backtracking...")
        bt_moves, bt_flags = self._run_backtracking()
        
        if bt_moves or bt_flags:
            if self.verbose: print(f"✨ BACKTRACKING SUCCÈS : {len(bt_moves)} sûres, {len(bt_flags)} mines identifiées par déduction complexe.")
            return list(bt_moves), list(bt_flags)
        
        if self.verbose: print("❌ Backtracking : Aucune certitude absolue trouvée (situation ambiguë).")

        # --- 3. PROBABILITÉS (Dernier recours) ---
        if self.verbose: print("🤔 Passage aux probabilités...")
        
        best_guess = self._get_safest_guess()
        if best_guess:
            moves.add(best_guess)
        
        return list(moves), list(flags)

    def _run_backtracking(self):
        """Teste toutes les combinaisons possibles sur la frontière"""
        boundary_vars = set()
        constraints = set()
        
        # Récupération de la frontière active
        for (x, y) in self.game.revealed:
            val = self.game.get_value(x, y)
            if val == 0: continue
            hidden = [n for n in self.game.get_neighbors(x, y) 
                      if n not in self.game.revealed and n not in self.game.flags]
            if hidden:
                for h in hidden: boundary_vars.add(h)
                constraints.add((x, y))
        
        boundary_list = list(boundary_vars)
        if not boundary_list: return [], []
        
        # Sécurité pour ne pas planter le PC
        if len(boundary_list) > self.MAX_BACKTRACK_VARS:
            if self.verbose: print(f"   -> Trop complexe ({len(boundary_list)} vars). Abandon du backtracking.")
            return [], []

        valid_solutions = []
        
        def solve_recursive(index, current_assignment):
            # Optimisation (Pruning)
            if not self._is_consistent(current_assignment, constraints): return

            if index == len(boundary_list):
                valid_solutions.append(current_assignment.copy())
                return

            var = boundary_list[index]
            
            # Hypothèse 0 : Pas de mine
            current_assignment[var] = 0
            solve_recursive(index + 1, current_assignment)
            
            # Hypothèse 1 : Mine
            current_assignment[var] = 1
            solve_recursive(index + 1, current_assignment)
            
            del current_assignment[var] # Backtrack

        solve_recursive(0, {})
        
        if not valid_solutions: return [], []

        if self.verbose: print(f"   -> {len(valid_solutions)} scénarios valides calculés.")

        # Analyse des résultats communs
        confirmed_mines = []
        confirmed_safe = []
        for var in boundary_list:
            is_always_mine = all(s[var] == 1 for s in valid_solutions)
            is_never_mine = all(s[var] == 0 for s in valid_solutions)
            
            if is_always_mine: confirmed_mines.append(var)
            elif is_never_mine: confirmed_safe.append(var)
            
        return confirmed_safe, confirmed_mines

    def _is_consistent(self, assignment, constraints):
        for (cx, cy) in constraints:
            val = self.game.get_value(cx, cy)
            neighbors = self.game.get_neighbors(cx, cy)
            
            mines_count = 0
            unknowns_count = 0
            
            for n in neighbors:
                if n in self.game.flags:
                    mines_count += 1
                elif n in assignment:
                    mines_count += assignment[n]
                elif n not in self.game.revealed:
                    unknowns_count += 1
            
            # Si on a déjà plus de mines que le chiffre -> Impossible
            if mines_count > val: return False
            # Si même avec toutes les inconnues on n'atteint pas le chiffre -> Impossible
            if mines_count + unknowns_count < val: return False
            
        return True

    def _get_safest_guess(self):
        prob_map = {}
        # Calcul simple de probabilité locale
        for (x, y) in self.game.revealed:
            val = self.game.get_value(x, y)
            neighbors = self.game.get_neighbors(x, y)
            hidden = [n for n in neighbors if n not in self.game.revealed and n not in self.game.flags]
            flagged = [n for n in neighbors if n in self.game.flags]
            
            if not hidden: continue
            
            # Formule : Mines restantes / Cases cachées
            prob = (val - len(flagged)) / len(hidden)
            
            for cell in hidden:
                # On garde la pire probabilité (le plus grand risque) pour une case donnée
                prob_map[cell] = max(prob_map.get(cell, 0), prob)

        # On sauvegarde la map pour que le GUI puisse la dessiner !
        self.game.prob_map = prob_map 

        if not prob_map:
            # Choix au hasard total (début de partie ou île isolée)
            hidden_cells = [(x,y) for x in range(self.game.width) for y in range(self.game.height) 
                           if (x,y) not in self.game.revealed and (x,y) not in self.game.flags]
            if hidden_cells:
                guess = random.choice(hidden_cells)
                if self.verbose: print(f"🎲 Aucune info : Tentative au hasard sur {guess}")
                return guess
            return None

        # Trier pour trouver le risque le plus faible
        sorted_guesses = sorted(prob_map.items(), key=lambda item: item[1])
        best_case = sorted_guesses[0][0]
        best_prob = sorted_guesses[0][1]
        
        if self.verbose: print(f"📊 Meilleure option : {best_case} avec {best_prob*100:.1f}% de risque.")
        return best_case