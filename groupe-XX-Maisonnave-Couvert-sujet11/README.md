💣 Démineur IA - Résolveur CSP & Probabiliste

Projet d'Intelligence Artificielle - ING4
Groupe XX - [Démineurs/Maisonnave-Couvert-De Gasquet]

Ce projet implémente un agent intelligent capable de résoudre le jeu du Démineur (Minesweeper) de manière autonome. Contrairement aux approches basiques, notre IA ne se contente pas de règles simples : elle modélise le jeu comme un **Problème de Satisfaction de Contraintes (CSP)**, utilise du **Backtracking** pour les cas complexes, et calcule des **probabilités** précises lorsque la logique pure ne suffit plus.


🚀 Installation et Lancement

Prérequis

* Python 3.8+
* Pygame

Installation

```bash
# Cloner le projet
git clone [URL_DU_REPO]
cd [NOM_DU_DOSSIER]

# Installer les dépendances
pip install pygame

```

commandes

Lancer le jeu (Interface Graphique) :
```bash
python src/main.py

```


Contrôles : Appuyez sur `R` à tout moment pour relancer une partie.


Lancer le Benchmark (Mode Performance) :
```bash
python src/benchmark.py

```


Note : Exécute 100+ simulations ultra-rapides sans affichage pour calculer le taux de victoire.


🧠 Notre Démarche : De la Naïveté à l'Expertise

Pour arriver à ce résultat, nous avons procédé par itérations successives, en augmentant progressivement l'intelligence de l'agent.

Phase 1 : L'approche Logique Déterministe (Single-Point)

Au début, nous avons implémenté les règles de base du Démineur. L'IA regardait une case et ses voisins.

 *Si `drapeaux == chiffre*` : Tout le reste est sûr.
 *Si `cases_cachées == chiffre*` : Tout le reste est une mine.
Limitation : Cette approche bloquait dès qu'il fallait déduire une information en croisant les données de plusieurs cases (ex: le motif "1-2-1").

Phase 2 : Modélisation CSP et Backtracking

Pour franchir un cap, nous avons considéré la "frontière" (les cases inconnues adjacentes aux cases révélées) comme un système d'équations.
Nous avons implémenté un algorithme de Backtracking (ou "Model Checking"). L'IA fait des hypothèses : *"Si je mets une mine ici, est-ce que ça rend le plateau impossible ?"*.
Si une hypothèse mène à une contradiction, nous savons avec certitude que la case est sûre.

Phase 3 : Gestion de l'Incertitude (Probabilités)

Parfois, le Démineur est purement une question de chance (50/50). Le backtracking ne trouvant aucune certitude, l'IA se bloquait.
Nous avons ajouté une couche probabiliste : si aucune logique ne fonctionne, l'IA calcule la probabilité de danger de chaque case frontière (`Mines Restantes / Cases Voisines`) et clique sur la case ayant le plus faible % de risque.


📂 Architecture du Code

Voici comment nous avons structuré le projet pour le rendre modulaire et maintenable.

1. `src/game_engine.py` (Le Moteur)

C'est le "Maître du Jeu". Il ne contient aucune intelligence, juste les règles.

- `Minesweeper.__init__` : Initialise la grille. Nous avons ajouté un booléen `first_click` pour garantir que le premier clic n'est jamais une mine (génération des mines *après* le premier clic).
- `reveal(x, y)` : Gère la logique de révélation et l'algorithme de "Flood Fill" (propagation) si on clique sur un 0.
- `get_neighbors(x, y)` : Utilitaire pour récupérer les coordonnées adjacentes valides.

2. `src/csp_solver.py` (Le Cerveau)

C'est le cœur de notre projet. La méthode `solve()` orchestre trois niveaux d'intelligence :

* Niveau 1 : `Logique Simple`
Parcourt toutes les cases révélées. Si une règle triviale s'applique, on l'exécute immédiatement. C'est très rapide (complexité O(N)).
* Niveau 2 : `_run_backtracking()`
Appelé quand le niveau 1 échoue.
1. Il isole les variables de la "frontière" (cases inconnues touchant des chiffres).
2. Il lance une récursion pour tester toutes les combinaisons valides de mines.
3. Optimisation : Si une case est une mine dans "tous" les scénarios valides, on la marque. Si elle est vide dans "tous" les scénarios, on la révèle.
4. Sécurité : Nous avons mis une limite (`MAX_BACKTRACK_VARS = 14`) pour éviter que l'arbre de récursion ne fasse geler l'ordinateur sur des situations trop complexes.


* Niveau 3 : `_get_safest_guess()`
Dernier recours. Calcule la probabilité locale de chaque case frontière. Retourne la case avec le score le plus bas. Permet aussi de dessiner la "Heatmap" de danger sur l'interface.

3. `src/gui.py` (L'Interface)

Gère l'affichage Pygame.

* `draw_probabilities()` : Une fonctionnalité clé. Elle récupère la `prob_map` générée par le solver et applique un calque de couleur (Vert -> Rouge) sur les cases pour visualiser "ce que l'IA pense".
* `draw_restart_overlay()` : Affiche le message de victoire/défaite avec un effet de clignotement pour inviter l'utilisateur à relancer.

4. `src/benchmark.py` (L'Audit)

Ce script est crucial pour valider notre projet. Il lance le jeu en mode "headless" (sans interface graphique) et coupe les `print` (mode silencieux).
Il nous permet d'affirmer objectivement : "Notre IA a un taux de réussite de X% sur 1000 parties".
A noter que lorsque on augmente la taille de la grille le taux de réussite diminue dratiquement mais sur une grille de 9x9 on obtients des scores bien mieux que ceux évoqués par une IA ou sur des sites, on est autour des 87% (sur 10 tests de 100 simulations). Quand on passe a une taille de 30x16 on passe souvent proche des 50%



📊 Performances et Résultats

Grâce à notre fichier `benchmark.py`, nous avons pu mesurer l'efficacité de l'IA sur une grille de taille 15x15 avec 30 mines (Difficulté Intermédiaire/Supérieure).

|  Type de Solveur  | Taux de Victoire (approx) |             Observations                  |
| ------------------| --------------------------| ------------------------------------------|
| Aléatoire         |         < 1%              |             Injouable.                    |
| Logique Simple    |         ~40%              | Bloque dès qu'un motif complexe apparaît. |
| CSP + Backtracking|         > 82%             |                   r*                      |

r* = Résout toutes les situations déductibles. Les défaites restantes sont dues aux situations de chance pure (50/50) inévitables au Démineur. (ca devenait trop compliqué d'écrire tout ce texte dans le tableau donc j'ai mis une astérisque)

✨ Fonctionnalités Bonus / Améliorations

* Safe Start : Impossible de perdre au premier clic (standard des versions modernes du jeu).
* Visualisation Debug : Affichage en temps réel des probabilités de danger sur la grille.
* Mode Silencieux : Le solver peut couper ses logs pour les tests de performance.
* Restart à chaud : Pas besoin de relancer le script, la touche 'R' réinitialise tout proprement.


👥 Auteurs

Projet réalisé dans le cadre du cours d'Intelligence Artificielle (ING4).

- [Gabriel] [Maisonnave] - Optimisation / Algorithme CSP
- [Raphael] [Couvert] - Interface Graphique / Moteur de jeu
- [Aurèle] [DeGasquet] - Benchmark & Moteur de Jeu 


Nous avons pris un réel plaisir à travailler sur ce sujet qui sort de l'ordinaire. Pour être totalement transparents, notre démarrage a été progressif : il nous a fallu un temps d'adaptation pour bien cerner les enjeux du sujet. Cependant, une fois la "machine lancée", l'investissement de l'équipe a été total et constant. Nous avons beaucoup appris, tant sur le plan technique qu'humain, et nous espérons que vous prendrez autant de plaisir à tester notre IA que nous en avons eu à la développer.


Fait avec ❤️ et beaucoup de café.
