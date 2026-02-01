# Wordle AI Solver (CSP + LLM)
Wordle est un jeu de réflexion basé sur les mots, où le but est de deviner un mot secret de 5 lettres en un nombre limité d’essais (généralement 6).

À chaque tentative, tu proposes un mot de 5 lettres, et le jeu te donne un retour lettre par lettre :

-   **Vert**  : la lettre est correcte et à la bonne position.
-   **Jaune**  : la lettre est dans le mot, mais pas à cette position.
-   **Gris**  : la lettre n’apparaît pas dans le mot (ou n’apparaît pas autant de fois, dans le cas des doublons).

Tu utilises ce feedback pour affiner tes essais suivants, éliminer des lettres impossibles, confirmer des positions, et converger vers le mot secret. Le point piège est la gestion des lettres répétées : une lettre peut être jaune/verte sur une occurrence et grise sur une autre si tu l’as proposée trop de fois par rapport au mot secret.

Nous avons créé un solveur Wordle hybride combinant :
- **CSP / filtrage exact** : applique strictement les règles Wordle pour éliminer les mots incompatibles.
- **LLM (Llama 3.1 via Ollama)** : propose un meilleur prochain guess parmi les candidats restants.

Le projet propose deux interfaces :
- **Streamlit (UI web)**
- **CLI (terminal)**


## Fonctionnement

À chaque tentative utilisateur, le solveur suit ces étapes :

1. **Entrée utilisateur**
   - Mode structuré : `GUESS FEEDBACK` (ex: `ORATE GVVJG`)
   - Mode texte libre : l’utilisateur décrit sa tentative ; un LLM extrait automatiquement un couple *(guess, feedback)*.

#### Format du feedback (V/J/G)
Le feedback est une chaîne de 5 caractères :
 - `V` = **Vert** (lettre correcte, bonne position)
 - `J` = **Jaune** (lettre présente mais mal placée)
 - `G` = **Gris** (lettre absente, ou excédentaire selon les doublons)

 2. **Filtrage CSP (contraintes Wordle)**
 - On calcule le feedback exact attendu pour chaque mot du dictionnaire.
 - On ne garde que les mots dont le feedback correspond exactement à l’historique de tentatives.
 - Gestion des lettres en double avec un comptage via Counter.

 3. **Ranking LLM**
   - Le LLM reçoit une liste de mots déjà validés par le CSP. Si beaucoup de candidats restent, on limite la liste envoyée au LLM.
   - Il retourne :
     - **Chosen word** : le mot recommandé à jouer maintenant
     - **Priority ranking (Top 3)** : un classement des 3 meilleurs mots parmi les candidats fournis,  pour donner d'autres id
 - Le LLM choisit uniquement parmi les candidats fournis.

## Prérequis
- Python 3.8+
- Télécharger [Ollama](https://ollama.com) 
- Installer le modèle llama3.1 :  ` ollama pull llama3.1`

## Installation

### 1) Cloner et créer un environnement virtuel

`python -m venv venv`

Activation Windows : `venv/bin/activate`

### 2) Installer les dépendances

`pip install streamlit keyboard ollama`

## Dictionnaire (wordle.txt)

Le solveur utilise un dictionnaire local `wordle.txt` :
- 1 mot par ligne
- exactement 5 lettres
- uniquement `A-Z` (pas d’apostrophe, pas de tiret, etc.)

Le projet a été testé avec ~22 000 mots anglais de 5 lettres.

## Lancer l’application Streamlit

`streamlit run app.py`

Dans l’UI :
- Choisis le mode d’entrée
- Clique Solve
- Possibilités de faire des tentatives successives
- Option Reset game pour repartir de zéro

## Lancer en CLI (terminal)

`python main.py`

Puis entrer des tentatives sous forme :
- `ORATE GVVJG`
- `ORATE -> GVVJG`

## Structure du projet

Le code est organisé autour de 3 modules logiques :

- `csp_solver.py`
  - `wordle_feedback_vjg(secret, guess)` : calcule le feedback exact
  - `solve_wordle_csp(dictionary, attempts)` : filtre les mots compatibles

- `llm_agent.py`
  - `_normalize_guess`, `_normalize_feedback` : validation
  - `extract_attempt_from_text(text)` : extraction via LLM (fallback)
  - `interroger_agent_wordle(prompt_utilisateur, dictionary_words, attempts)` : pipeline complet

- `app.py` (Streamlit)
- `main.py` (CLI)
- `wordle.txt` (dictionnaire)

## Limitations

- Le LLM ne “devine” pas le mot secret : il choisit un guess parmi les candidats restants.
- En mode texte libre, l’extraction dépend de la qualité du prompt et du modèle.
- Si aucune solution n’est trouvée, cela indique généralement :
  - une erreur de saisie dans le feedback,
  - un dictionnaire incomplet / incompatible,
  - ou un historique de tentatives incohérent.
 
## Slides
https://www.canva.com/design/DAG-755-SdM/nmlLr8XF8taDIG-l5zZFVQ/edit?utm_content=DAG-755-SdM&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

