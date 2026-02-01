import sys
import os
import random
import threading
import webbrowser
import json
import logging

# Vérification de Flask
try:
    from flask import Flask, render_template_string, jsonify, request
except ImportError:
    print("ERREUR : La bibliothèque 'flask' est manquante.")
    print("Veuillez l'installer avec la commande : pip install flask")
    sys.exit(1)

from grid_structure import GridStructure
from solveur import CrosswordSolver, PATH_DICO

try:
    from ortools.sat.python import cp_model
except ImportError:
    cp_model = None

# --- CONFIGURATION ---
ROWS = 12
COLS = 12
NB_NOIRES = 26

# --- INITIALISATION FLASK ---
app = Flask(__name__)

# Désactivation des logs verbeux de Flask (supprime le warning "Development Server")
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# --- LOGIQUE DE GÉNÉRATION (OR-TOOLS) ---
# Cette fonction utilise la programmation par contraintes (CP-SAT) pour créer une grille valide.
def generate_grid_logic():
    if cp_model is None:
        return [['.' for _ in range(COLS)] for _ in range(ROWS)]

    model = cp_model.CpModel()
    grid_vars = {}

    for r in range(ROWS):
        for c in range(COLS):
            grid_vars[(r, c)] = model.NewBoolVar(f'c_{r}_{c}')

    # 1. Contrainte : Nombre exact de cases noires
    model.Add(sum(grid_vars.values()) == NB_NOIRES)

    # 2. Contrainte : Pas de mots de 1 lettre (chaque case blanche doit avoir une voisine blanche)
    for r in range(ROWS):
        for c in range(COLS):
            left = grid_vars[(r, c-1)] if c > 0 else 1
            right = grid_vars[(r, c+1)] if c < COLS - 1 else 1
            top = grid_vars[(r-1, c)] if r > 0 else 1
            bottom = grid_vars[(r+1, c)] if r < ROWS - 1 else 1
            model.Add(left + right < 2).OnlyEnforceIf(grid_vars[(r, c)].Not())
            model.Add(top + bottom < 2).OnlyEnforceIf(grid_vars[(r, c)].Not())

    # 3. Contrainte : Symétrie centrale (Rotation 180°) pour l'esthétique
    for r in range(ROWS):
        for c in range(COLS):
            model.Add(grid_vars[(r, c)] == grid_vars[(ROWS - 1 - r, COLS - 1 - c)])

    # 4. Contrainte : Éviter les gros blocs de cases noires (Max 2 consécutives)
    for r in range(ROWS):
        for c in range(COLS - 2):
            model.Add(grid_vars[(r, c)] + grid_vars[(r, c+1)] + grid_vars[(r, c+2)] <= 2)
    for c in range(COLS):
        for r in range(ROWS - 2):
            model.Add(grid_vars[(r, c)] + grid_vars[(r+1, c)] + grid_vars[(r+2, c)] <= 2)

    # 5. Contrainte : Pas de mots de 12 lettres (taille max de la grille)
    # On impose au moins une case noire par ligne et par colonne.
    for r in range(ROWS):
        model.Add(sum(grid_vars[(r, c)] for c in range(COLS)) >= 1)
    for c in range(COLS):
        model.Add(sum(grid_vars[(r, c)] for r in range(ROWS)) >= 1)

    # Objectif : Introduire de l'aléatoire pour avoir des grilles différentes à chaque fois
    objective_terms = []
    for r in range(ROWS):
        for c in range(COLS):
            objective_terms.append(grid_vars[(r, c)] * random.randint(1, 100))
    model.Minimize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.random_seed = random.randint(0, 2**30)
    status = solver.Solve(model)

    # Reconstruction de la grille sous forme de liste de listes
    new_grid = [['.' for _ in range(COLS)] for _ in range(ROWS)]
    if status == cp_model.FEASIBLE or status == cp_model.OPTIMAL:
        for r in range(ROWS):
            for c in range(COLS):
                if solver.Value(grid_vars[(r, c)]) == 1:
                    new_grid[r][c] = '#'
    return new_grid

# --- PAGE WEB (HTML/CSS/JS) ---
HTML_PAGE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Mots Croisés IA</title>
    <style>
        body { font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f8f9fa; margin: 0; padding: 0; color: #333; }
        header { background-color: #fff; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }
        h1 { margin: 0; color: #4285f4; font-size: 24px; }
        
        .container { max-width: 1100px; margin: 20px auto; background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24); overflow: hidden; }
        
        /* Onglets */
        .tabs { display: flex; border-bottom: 1px solid #e0e0e0; background: #fafafa; }
        .tab-btn { flex: 1; padding: 15px; border: none; background: none; cursor: pointer; font-size: 14px; font-weight: 500; color: #5f6368; text-transform: uppercase; transition: background 0.3s; outline: none; }
        .tab-btn:hover { background: #f1f3f4; color: #202124; }
        .tab-btn.active { border-bottom: 3px solid #4285f4; color: #4285f4; font-weight: bold; background: white; }
        
        .tab-content { display: none; padding: 30px; min-height: 500px; }
        .tab-content.active { display: block; }
        
        /* Grille */
        .grid-container { display: flex; justify-content: center; margin: 20px 0; }
        table { border-collapse: collapse; border: 2px solid #333; }
        td { width: 40px; height: 40px; border: 1px solid #999; text-align: center; font-weight: bold; font-size: 20px; position: relative; padding: 0; }
        .black { background-color: #202124; }
        .white { background-color: white; color: #202124; }
        
        /* Inputs pour le jeu */
        .cell-input {
            width: 100%; height: 100%; border: none; background: transparent;
            text-align: center; font-size: 20px; font-weight: bold; color: #202124;
            text-transform: uppercase; padding: 0; margin: 0;
        }
        .cell-input:focus { outline: none; background-color: #e8f0fe; }
        .correct { background-color: #a5d6a7 !important; } /* Vert pour juste */
        
        /* Boutons */
        .btn { background-color: #1a73e8; color: white; border: none; padding: 10px 24px; border-radius: 4px; font-size: 14px; font-weight: 500; cursor: pointer; transition: background 0.2s; box-shadow: 0 1px 2px rgba(0,0,0,0.2); }
        .btn:hover { background-color: #1765cc; box-shadow: 0 1px 3px rgba(0,0,0,0.3); }
        .btn:disabled { background-color: #ccc; cursor: default; }
        
        /* Listes */
        .lists-wrapper { display: flex; gap: 20px; margin-top: 20px; }
        .list-col { flex: 1; }
        .list-col h3 { font-size: 16px; color: #202124; border-bottom: 2px solid #eee; padding-bottom: 10px; }
        ul { list-style: none; padding: 0; max-height: 400px; overflow-y: auto; }
        li { padding: 8px 0; border-bottom: 1px solid #f1f3f4; font-size: 14px; }
        li strong { color: #1a73e8; margin-right: 5px; }
        
        #status-msg { margin-top: 15px; font-weight: 500; min-height: 20px; }
        #game-msg { color: #d93025; font-weight: bold; margin-bottom: 10px; min-height: 24px; text-align: center; }
    </style>
</head>
<body>
    <header>
        <h1>Générateur de Mots Croisés - IA</h1>
    </header>

    <div class="container">
        <div class="tabs">
            <button class="tab-btn active" onclick="openTab('tab1')">1. Structure</button>
            <button class="tab-btn" onclick="openTab('tab2')">2. Résolution</button>
            <button class="tab-btn" onclick="openTab('tab3')">3. Mode Jeu</button>
            <button class="tab-btn" onclick="openTab('tab4')">4. Solution</button>
        </div>

        <!-- ONGLET 1 : STRUCTURE -->
        <div id="tab1" class="tab-content active">
            <div style="text-align:center;">
                <p>Générez une nouvelle structure de grille symétrique.</p>
                <button class="btn" onclick="generateGrid()">Nouvelle Structure Aléatoire</button>
                <div id="grid-view-1" class="grid-container"></div>
            </div>
        </div>

        <!-- ONGLET 2 : RÉSOLUTION -->
        <div id="tab2" class="tab-content">
            <div style="text-align:center; margin-top: 50px;">
                <h2>Lancer l'Intelligence Artificielle</h2>
                <p>L'IA va chercher à remplir la grille générée avec les mots du dictionnaire.</p>
                <br>
                <button id="btn-solve" class="btn" onclick="solveGrid()">Lancer la Résolution</button>
                <p id="status-msg"></p>
            </div>
        </div>

        <!-- ONGLET 3 : JEU (NOUVEAU) -->
        <div id="tab3" class="tab-content">
            <div style="text-align: center;">
                <div id="game-msg"></div>
                <div id="word-counter" style="font-size: 18px; font-weight: bold; color: #1a73e8; margin-bottom: 10px;"></div>
                <button id="btn-dir" class="btn" style="margin-bottom: 15px; background-color: #fbbc04; color: #202124;" onclick="toggleDirection()">Direction : Horizontale (➡)</button>
            </div>
            <div style="display: flex; gap: 40px;">
                <div style="flex: 0 0 auto;">
                    <div id="grid-view-game" class="grid-container"></div>
                </div>
                <div style="flex: 1;">
                    <div class="lists-wrapper">
                        <div class="list-col">
                            <h3>Horizontalement</h3>
                            <ul id="list-h-game"></ul>
                        </div>
                        <div class="list-col">
                            <h3>Verticalement</h3>
                            <ul id="list-v-game"></ul>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- ONGLET 4 : SOLUTION -->
        <div id="tab4" class="tab-content">
            <div style="display: flex; gap: 40px;">
                <div style="flex: 0 0 auto;">
                    <div id="grid-view-sol" class="grid-container"></div>
                </div>
                <div style="flex: 1;">
                    <div class="lists-wrapper">
                        <div class="list-col">
                            <h3>Horizontalement</h3>
                            <ul id="list-h-sol"></ul>
                        </div>
                        <div class="list-col">
                            <h3>Verticalement</h3>
                            <ul id="list-v-sol"></ul>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentGrid = [];
        let gameDefinitions = []; // Stocke les solutions pour la validation
        let inputDirection = 'H'; // Direction actuelle de saisie ('H' ou 'V')
        let foundWords = new Set(); // Mots déjà trouvés par le joueur

        // Bascule la direction de saisie (Horizontal <-> Vertical)
        function toggleDirection() {
            inputDirection = inputDirection === 'H' ? 'V' : 'H';
            const btn = document.getElementById('btn-dir');
            if (inputDirection === 'H') {
                btn.innerText = "Direction : Horizontale (➡)";
            } else {
                btn.innerText = "Direction : Verticale (⬇)";
            }
        }

        // Gestion des onglets (Tabs)
        function openTab(tabName) {
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.getElementById(tabName).classList.add('active');
            
            const index = ['tab1', 'tab2', 'tab3', 'tab4'].indexOf(tabName);
            document.querySelectorAll('.tab-btn')[index].classList.add('active');
        }

        // Dessine la grille HTML selon le mode
        // mode: 'view' (simple), 'input' (jeu), 'solution' (réponse affichée)
        function drawGrid(containerId, gridData, mode = 'view', solutionData = null) {
            const container = document.getElementById(containerId);
            let html = '<table>';
            
            // En-têtes colonnes
            html += '<tr><td></td>';
            for(let c=0; c<gridData[0].length; c++) {
                html += `<td style="background:#eee; font-size:12px; border:none;">${String.fromCharCode(65+c)}</td>`;
            }
            html += '</tr>';

            for(let r=0; r<gridData.length; r++) {
                html += '<tr>';
                html += `<td style="background:#eee; font-size:12px; border:none;">${r+1}</td>`;
                for(let c=0; c<gridData[r].length; c++) {
                    let cellClass = gridData[r][c] === '#' ? 'black' : 'white';
                    let content = '';
                    
                    if (mode === 'solution' && solutionData && gridData[r][c] !== '#') {
                        content = solutionData[r][c] || '';
                    } else if (mode === 'input' && gridData[r][c] !== '#') {
                        // Mode Jeu : Input
                        content = `<input type="text" class="cell-input" maxlength="1" 
                                   data-r="${r}" data-c="${c}" oninput="handleInput(this)" onkeydown="handleKeyDown(event, this)">`;
                    }
                    
                    // ID unique pour chaque case en mode jeu pour pouvoir changer la couleur
                    let tdId = mode === 'input' ? `cell-${r}-${c}` : '';
                    
                    html += `<td id="${tdId}" class="${cellClass}">${content}</td>`;
                }
                html += '</tr>';
            }
            html += '</table>';
            container.innerHTML = html;
        }

        // Appel API pour générer une nouvelle structure de grille
        function generateGrid() {
            fetch('/api/generate')
                .then(res => res.json())
                .then(data => {
                    currentGrid = data.grid;
                    drawGrid('grid-view-1', currentGrid, 'view');
                    
                    // Reset des autres onglets
                    document.getElementById('grid-view-game').innerHTML = '';
                    document.getElementById('grid-view-sol').innerHTML = '';
                    document.getElementById('list-h-game').innerHTML = '';
                    document.getElementById('list-v-game').innerHTML = '';
                    document.getElementById('list-h-sol').innerHTML = '';
                    document.getElementById('list-v-sol').innerHTML = '';
                    document.getElementById('status-msg').innerText = '';
                    document.getElementById('btn-solve').disabled = false;
                });
        }

        // Appel API pour résoudre la grille avec l'IA
        function solveGrid() {
            const btn = document.getElementById('btn-solve');
            const status = document.getElementById('status-msg');
            
            btn.disabled = true;
            status.innerText = "Recherche en cours... (Cela peut prendre quelques secondes)";
            status.style.color = "#1a73e8";
            
            fetch('/api/solve', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({grid: currentGrid})
            })
            .then(res => res.json())
            .then(data => {
                btn.disabled = false;
                if(data.success) {
                    status.innerText = "Solution trouvée !";
                    status.style.color = "green";
                    
                    // Stocker les définitions pour le jeu
                    gameDefinitions = data.definitions;
                    foundWords.clear();
                    updateCounter();

                    // 1. Préparer l'onglet JEU (Tab 3)
                    drawGrid('grid-view-game', currentGrid, 'input');
                    fillDefinitions('list-h-game', 'list-v-game', data.definitions, false); // false = cacher réponse

                    // 2. Préparer l'onglet SOLUTION (Tab 4)
                    drawGrid('grid-view-sol', currentGrid, 'solution', data.letter_grid);
                    fillDefinitions('list-h-sol', 'list-v-sol', data.definitions, true); // true = montrer réponse
                    
                    // Basculer vers l'onglet Jeu
                    setTimeout(() => openTab('tab3'), 500);
                } else {
                    status.innerText = "Impossible de résoudre cette grille.";
                    status.style.color = "red";
                }
            })
            .catch(err => {
                btn.disabled = false;
                status.innerText = "Erreur technique.";
                status.style.color = "red";
                console.error(err);
            });
        }

        // Remplit les listes de définitions dans l'interface
        function fillDefinitions(idH, idV, defs, showWord) {
            const listH = document.getElementById(idH);
            const listV = document.getElementById(idV);
            listH.innerHTML = '';
            listV.innerHTML = '';
            
            defs.forEach(item => {
                const li = document.createElement('li');
                let text = `<strong>${item.coord}</strong> ${item.def}`;
                if (showWord) {
                    text += ` <span style="color:#999">(${item.word})</span>`;
                }
                li.innerHTML = text;
                if(item.dir === 'H') listH.appendChild(li);
                else listV.appendChild(li);
            });
        }

        // Met à jour le compteur de mots trouvés
        function updateCounter() {
            const counter = document.getElementById('word-counter');
            if (gameDefinitions.length > 0) {
                counter.innerText = `Mots trouvés : ${foundWords.size} / ${gameDefinitions.length}`;
            } else {
                counter.innerText = "";
            }
        }

        // --- LOGIQUE DU JEU ---

        // Déplace le focus vers la case suivante selon la direction
        function focusNext(r, c, dir) {
            let nextR = r;
            let nextC = c;
            if (dir === 'H') nextC++;
            else nextR++;
            
            let nextInput = document.querySelector(`#cell-${nextR}-${nextC} input`);
            if (nextInput) {
                nextInput.focus();
            }
        }

        // Gère la saisie d'une lettre dans une case
        function handleInput(input) {
            // 1. Nettoyage : Lettres majuscules uniquement
            input.value = input.value.replace(/[^a-zA-Z]/g, '').toUpperCase();

            // 2. Navigation automatique
            if (input.value.length > 0) {
                let r = parseInt(input.getAttribute('data-r'));
                let c = parseInt(input.getAttribute('data-c'));
                
                focusNext(r, c, inputDirection);
            }
            
            // 3. Vérification des mots
            checkWords();
        }

        // Gère les touches spéciales (Flèches, Backspace)
        function handleKeyDown(e, input) {
            let r = parseInt(input.getAttribute('data-r'));
            let c = parseInt(input.getAttribute('data-c'));

            // 1. Navigation avec les flèches (saut des cases noires)
            if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key)) {
                e.preventDefault();
                let dR = 0, dC = 0;
                if (e.key === 'ArrowUp') dR = -1;
                else if (e.key === 'ArrowDown') dR = 1;
                else if (e.key === 'ArrowLeft') dC = -1;
                else if (e.key === 'ArrowRight') dC = 1;

                let currR = r + dR;
                let currC = c + dC;
                
                while (true) {
                    let cell = document.getElementById(`cell-${currR}-${currC}`);
                    if (!cell) break; // Hors grille
                    
                    let nextInput = cell.querySelector('input');
                    if (nextInput) {
                        nextInput.focus();
                        break;
                    }
                    // Case noire -> on continue
                    currR += dR;
                    currC += dC;
                }
            }
            // 2. Effacer (Backspace)
            else if (e.key === 'Backspace') {
                e.preventDefault();
                
                // 1. Effacer le contenu si présent et non verrouillé
                if (!input.readOnly && input.value !== '') {
                    input.value = '';
                    checkWords();
                }
                
                // 2. Reculer (Gauche ou Haut)
                if (inputDirection === 'H') c--;
                else r--;
                
                // Focus sur la case précédente si elle existe
                let prevInput = document.querySelector(`#cell-${r}-${c} input`);
                if (prevInput) prevInput.focus();
            }
            // 3. Saisie sur case validée (ReadOnly) -> Avancer seulement
            else if (input.readOnly && e.key.length === 1 && e.key.match(/[a-zA-Z]/)) {
                e.preventDefault(); // On empêche l'écriture de la lettre
                // On passe simplement à la case suivante
                focusNext(r, c, inputDirection);
            }
            // 4. Saisie sur case normale (Overwrite sans sélection)
            else if (!input.readOnly && !e.ctrlKey && !e.metaKey && e.key.length === 1 && e.key.match(/[a-zA-Z]/)) {
                e.preventDefault();
                input.value = e.key.toUpperCase();
                handleInput(input);
            }
        }

        // Vérifie si les mots saisis correspondent à la solution
        function checkWords() {
            const msgDiv = document.getElementById('game-msg');
            msgDiv.innerText = ""; // Reset message
            let errorFound = false;
            let newWordFound = false;

            // Pour chaque mot de la solution
            gameDefinitions.forEach(def => {
                // Calculer les coordonnées des cases du mot
                // Coord est du type "A1", "B12"
                let colChar = def.coord.charAt(0);
                let rowStr = def.coord.substring(1);
                
                let startCol = colChar.charCodeAt(0) - 65;
                let startRow = parseInt(rowStr) - 1;
                let length = def.word.length;
                
                let userWord = "";
                let isComplete = true;
                let cellIds = [];

                for(let i=0; i<length; i++) {
                    let r = startRow;
                    let c = startCol;
                    if (def.dir === 'H') c += i;
                    else r += i;
                    
                    let input = document.querySelector(`#cell-${r}-${c} input`);
                    if (input && input.value) {
                        userWord += input.value;
                    } else {
                        isComplete = false;
                    }
                    cellIds.push(`cell-${r}-${c}`);
                }

                // Si le mot est complet
                if (isComplete) {
                    if (userWord === def.word) {
                        // Mot juste -> Vert
                        cellIds.forEach(id => {
                            let cell = document.getElementById(id);
                            cell.classList.add('correct');
                            let inp = cell.querySelector('input');
                            if (inp) inp.readOnly = true;
                        });
                        
                        let key = def.coord + '-' + def.dir;
                        if (!foundWords.has(key)) {
                            foundWords.add(key);
                            newWordFound = true;
                            updateCounter();
                        }
                    } else {
                        // Mot faux -> Message
                        errorFound = true;
                    }
                }
            });

            if (newWordFound) {
                msgDiv.innerText = "Vous avez trouvé le mot !";
                msgDiv.style.color = "green";
            } else if (errorFound) {
                msgDiv.innerText = "Ce n'est pas le mot";
                msgDiv.style.color = "#d93025";
            }
        }

        // Init
        generateGrid();
    </script>
</body>
</html>
"""

# --- ROUTES FLASK ---

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

# API : Génère la structure (cases noires/blanches)
@app.route('/api/generate')
def api_generate():
    new_grid = generate_grid_logic()
    return jsonify({"grid": new_grid})

# API : Résout la grille avec les mots du dictionnaire
@app.route('/api/solve', methods=['POST'])
def api_solve():
    data = request.get_json()
    grid_data = data.get('grid')
    grid_strings = ["".join(row) for row in grid_data]
    
    # On appelle le solveur en lui disant de NE PAS générer le HTML (render_html=False)
    solver = CrosswordSolver(grid_strings, PATH_DICO)
    solver.solve(render_html=False) 
    
    if solver.solution:
        letter_grid = [['' for _ in range(COLS)] for _ in range(ROWS)]
        for slot_id, word in solver.solution.items():
            slot = next(s for s in solver.structure.slots if s.id == slot_id)
            for i, (r, c) in enumerate(slot.cells):
                letter_grid[r][c] = word[i]
        
        defs_list = []
        sorted_slots = sorted(solver.structure.slots, key=lambda s: (s.direction, s.row, s.col) if s.direction == 'H' else (s.direction, s.col, s.row))
        for s in sorted_slots:
            if s.id in solver.solution:
                word = solver.solution[s.id]
                defn = solver.definitions.get(word, "Pas de définition")
                coord = f"{chr(65+s.col)}{s.row+1}"
                defs_list.append({"dir": s.direction, "coord": coord, "word": word, "def": defn})
        
        return jsonify({"success": True, "letter_grid": letter_grid, "definitions": defs_list})
    else:
        return jsonify({"success": False})

# Ouvre le navigateur automatiquement
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

# --- POINT D'ENTRÉE PRINCIPAL ---
if __name__ == "__main__":
    threading.Timer(1.5, open_browser).start()
    print("Démarrage du serveur web...")
    app.run(debug=False, port=5000)
