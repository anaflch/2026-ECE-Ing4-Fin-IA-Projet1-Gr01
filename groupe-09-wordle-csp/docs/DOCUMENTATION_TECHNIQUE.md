# Documentation technique – Solveur Wordle (CSP + LLM)

## 1. Objectif

Ce projet vise à assister la résolution de Wordle (5 lettres) en combinant :

- un **filtrage déterministe** basé sur les règles exactes du jeu (formalisé comme un problème de contraintes),
- un **LLM** (via Ollama) utilisé uniquement pour :
  1) extraire une tentative depuis du texte libre (fallback),
  2) proposer un classement Top 3 des meilleurs mots parmi les candidats déjà validés.

Le LLM n’est jamais la “source de vérité” : le CSP (filtrage exact) décide quels mots sont possibles.

## 2. Structure du projet
Arborescence (exemple) :

- `src/`
  - `csp_solver.py` : règles Wordle + filtrage des candidats
  - `llm_agent.py` : orchestration (parsing, extraction LLM, ranking LLM)
  - `app.py` : UI Streamlit
  - `main.py` : interface CLI
  - `wordle.txt` : dictionnaire (mots 5 lettres)
- `docs/`
  - `DOCUMENTATION_TECHNIQUE.md`

> Hypothèse : `wordle.txt` contient des mots anglais en A–Z uniquement.  
> La notation `V/J/G` est une convention interne (Vert/Jaune/Gris).

## 3. Modélisation du Wordle comme CSP

### 3.1 Domaine
- Un mot secret est une chaîne de longueur 5 : $$ w \in \{A..Z\}^5 $$.
- Le dictionnaire est l’ensemble fini des valeurs possibles.

### 3.2 Contraintes
Chaque tentative ajoute une contrainte de la forme :

- donnée une proposition `guess` (5 lettres),
- et un feedback `fb` (5 caractères parmi `V`, `J`, `G`),

Le mot secret `w` est valide si :

- `wordle_feedback_vjg(w, guess) == fb`.

Cette formulation a deux avantages :
- elle respecte automatiquement les règles exactes, y compris les doublons,
- elle reste simple : on n’encode pas manuellement les contraintes, on compare un feedback calculé.

## 4. Module `csp_solver.py`

### 4.1 `wordle_feedback_vjg(secret: str, guess: str) -> str`

**But :** calculer le feedback `V/J/G` selon les règles Wordle.

**Principe :**
1. Marquer les verts (lettre correcte à la bonne position).
2. Pour les positions restantes, marquer les jaunes uniquement si la lettre existe encore dans le mot secret en tenant compte du nombre d’occurrences (gestion des doublons via un compteur).

**Invariants :**
- `secret` et `guess` sont normalisés en majuscules.
- Si les longueurs ne valent pas 5 → `ValueError`.

**Exemple :**
- `secret="APPLE"`, `guess="ALLEY"`  
  Le comptage empêche d’attribuer plus de jaunes que d’occurrences disponibles.

### 4.2 `solve_wordle_csp(possible_words, attempts) -> list[str]`

**But :** filtrer le dictionnaire en ne conservant que les mots compatibles avec l’historique des tentatives.

**Entrées :**
- `possible_words` : itérable de mots candidats (typiquement le dictionnaire complet)
- `attempts` : liste `[(guess, feedback), ...]`

**Sortie :**
- liste des mots `w` tels que, pour toute tentative, le feedback calculé correspond exactement au feedback attendu.

**Complexité :**
- Temps ≈ $$ O(N \times A \times 5) $$ où `N`=taille du dictionnaire, `A`=nombre de tentatives.
- Mémoire : faible (liste des solutions + structures temporaires).


## 5. Module `llm_agent.py`

### 5.1 Parsing & normalisation

Le module définit des fonctions de validation :
- `_normalize_guess(s)` : vérifie chaîne, longueur 5, lettres A–Z
- `_normalize_feedback(s)` : vérifie longueur 5, alphabet `{V,J,G}`

Un regex `_DIRECT` accepte notamment :
- `ORATE GVVJG`
- `ORATE->GVVJG`
- `ORATE -> GVVJG`

### 5.2 Extraction depuis texte libre (fallback LLM)

`extract_attempt_from_text(user_text) -> Optional[dict]`

**But :**
- Lorsque l’entrée n’est pas au format direct, demander au LLM d’extraire exactement une tentative `(guess, feedback)`.

**Contrôle :**
- Si l’extraction échoue ou si les champs ne respectent pas les validateurs → `None`.

> Remarque sécurité/robustesse : même si le LLM renvoie n’importe quoi, la normalisation empêche d’ajouter une tentative invalide.

### 5.3 Orchestration complète

`interroger_agent_wordle(prompt_utilisateur, dictionary_words, attempts) -> str`

Pipeline :
1. Parsing direct ou extraction LLM
2. Ajout à l’historique `attempts.append((guess, feedback))`
3. Filtrage CSP : `possible = solve_wordle_csp(...)`
4. Si `possible` vide → message d’erreur (contraintes incohérentes)
5. **Ranking LLM** :
   - on envoie au LLM une liste limitée de candidats (`MAX_CANDIDATES_TO_LLM`, ex. 40),
   - le LLM doit choisir uniquement dans cette liste et renvoyer :
     - `Chosen word: <WORD>`
     - `Priority ranking:` (Top 3)

## 6. Limitation / contrôle du LLM

Objectifs :
- réduire la latence et le coût,
- éviter de “noyer” le modèle avec trop de candidats.

Stratégie actuelle :
- si trop de solutions : sélection d’un sous-ensemble, trié par diversité (`len(set(w))`), puis tronqué.

Propriété clé :
- **le CSP reste la source de vérité** ; le LLM ne fait que prioriser.


## 7. Interfaces

### 7.1 CLI (`main.py`)
- boucle interactive
- historique conservé pendant l’exécution
- affichage : tentative ajoutée, candidats, décision/ranking LLM

### 7.2 Web UI (`app.py`, Streamlit)
- mode structuré (guess + feedback)
- mode texte libre (extraction LLM)
- table d’historique, bouton reset, affichage résultat


## 8. Exécution

### 8.1 Prérequis
- Python 3.8+ selon ton choix
- Ollama installé
- Modèle : `llama3.1`

### 8.2 Installer
`pip install streamlit keyboard ollama`

### 8.3 Lancer la UI Streamlit
`streamlit run src/app.py`

### 8.4 Lancer en CLI
`python src/main.py`


## 9. Dépannage (troubleshooting)

- **“No solution matches the current constraints”** :
  - feedback saisi incorrect,
  - tentative mal retranscrite,
  - dictionnaire ne contient pas le mot secret.
- **Extraction texte libre qui échoue** :
  - le texte ne contient pas clairement un guess 5 lettres et un feedback V/J/G,
  - reformuler en `ORATE -> GVVJG`.
- **Problèmes avec `keyboard`** :
  - dépend de l’OS / permissions ; optionnel si tu acceptes de quitter via Ctrl+C.



