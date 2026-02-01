# Documentation Technique - Projet Mots Croisés

## 1. Contexte et Sujet
**Sujet 17 : Construction de mots-croisés par contraintes.**

Ce projet modélise la génération et la résolution de grilles comme un **Problème de Satisfaction de Contraintes (CSP)**. L'objectif est de remplir une grille noire/blanche avec des mots qui se croisent de façon cohérente (les lettres aux intersections doivent être identiques).

### Références Théoriques
*   **Approche utilisée** : Programmation par Contraintes (CP) sur des domaines finis.
*   **Inspiration** : *Generating Crossword Grids Using Constraint Programming* (Guide OR-Tools) et travaux de G. Gervet (1995) sur la logique de contraintes.

---

## 2. Architecture du Code

Le projet est structuré autour de trois composants principaux :

### A. L'Interface (`interface_graphique.py`)
*   **Rôle** : Serveur Web (Flask) et Interface Utilisateur.
*   **Fonctionnement** :
    *   Gère les routes API (`/api/generate`, `/api/solve`).
    *   Contient le modèle CP-SAT pour la **génération de la structure** (placement des cases noires selon des règles de symétrie et de connectivité).

### B. Le Solveur (`solveur.py`)
*   **Rôle** : Le "cerveau" de l'IA.
*   **Technologie** : Google OR-Tools (module `cp_model`).
*   **Algorithme** :
    1.  Charge le dictionnaire en mémoire.
    2.  Crée une variable pour chaque case de la grille (A-Z).
    3.  Applique la contrainte `AddAllowedAssignments` (Table Constraint) pour chaque emplacement de mot : la suite de lettres doit former un mot valide du dictionnaire.
    4.  Le solveur propage les contraintes pour trouver une solution valide.

### C. L'Analyseur (`grid_structure.py`)
*   **Rôle** : Outil d'analyse géométrique.
*   **Fonctionnement** : Scanne la grille (matrice de caractères) pour identifier les "slots" (emplacements de mots horizontaux et verticaux) et leurs intersections.

---

## 3. Données et Dictionnaire

*   **Fichier source** : `fichier_texte/donne_reponse/dico_definitions_organise.txt`
*   **Format** : Les mots sont triés par longueur pour optimiser la recherche.
*   **Nettoyage** : Un script (`formatage_definitions.py`) a été utilisé pour nettoyer le fichier brut (suppression des accents, correction des erreurs OCR comme "ELL" -> "ELLE").

---

## 4. Fonctionnalités de l'Application

1.  **Onglet Structure** : Génération procédurale de grilles valides (symétrie centrale, pas de mots de 1 lettre).
2.  **Onglet Résolution** : L'IA remplit la grille. Si aucune solution n'existe avec le dictionnaire actuel, elle l'indique.
3.  **Onglet Jeu** :
    *   Mode interactif pour l'utilisateur.
    *   Validation en temps réel.
    *   Navigation fluide (saut automatique des cases noires, écrasement des lettres).
4.  **Onglet Solution** : Affiche la grille complète résolue.

---

## 5. Comparaison : Pourquoi utiliser OR-Tools ?

C'est vraiment là que tout se joue. Pour bien comprendre l'intérêt de l'IA, comparons une méthode classique (force brute) avec notre solution par contraintes.

### A. L'Approche Naïve (Backtracking simple)
*Le "Test bête et méchant".*
L'algorithme essaie un mot, passe au suivant, et s'il se trompe, il revient en arrière.

**Scénario** : On cherche deux mots qui se croisent sur la 2ème lettre.
*   Horizontal : _ _ _ (3 lettres)
*   Vertical : _ _ _ (3 lettres)

```text
1. Essai H : "BUS"
    2. Essai V : "CAR" (Croisement 'U' vs 'A' ?) -> ÉCHEC
    2. Essai V : "NET" (Croisement 'U' vs 'E' ?) -> ÉCHEC
    2. Essai V : "SOL" (Croisement 'U' vs 'O' ?) -> ÉCHEC
    ... (Teste tous les mots du dictionnaire pour V)
3. Retour arrière (Backtrack)
1. Essai H : "BUT"
    2. Essai V : "CAR" ... -> ÉCHEC
```
**Résultat** : Explosion combinatoire. Des millions de tests inutiles.

### B. L'Approche OR-Tools (Propagation de Contraintes)
*L'intelligence artificielle.*
L'algorithme réduit le champ des possibles **avant** de tester.

**Même scénario** :
1.  **Contrainte** : La 2ème lettre de H doit être égale à la 2ème lettre de V.
2.  **Propagation** :
    *   Si je mets "BUS" en H, alors V *doit* avoir un 'U' en 2ème position.
    *   L'IA supprime *immédiatement* "CAR", "NET", "SOL" de la liste des candidats pour V.
    *   Elle ne garde que les mots compatibles (ex: "PUR", "DUR").

```text
Domaine H : {BUS, BUT, CAR, SEL}
Domaine V : {PUR, DUR, CAR, NET}

1. Propagation : "CAR" (H) et "SEL" (H) sont éliminés car aucun mot de V n'a 'A' ou 'E' en 2ème position.
2. Solution directe : H="BUS" ou "BUT", V="PUR" ou "DUR".
```
**Résultat** : Résolution quasi-instantanée sans tests inutiles.

---

## 6. Pistes d'Amélioration

*   **Optimisation** : Utiliser un dictionnaire sous forme d'arbre (Trie) ou de DAWG pour réduire l'empreinte mémoire.
*   **Thématiques** : Filtrer le dictionnaire pour ne générer que des grilles sur un thème précis (ex: Géographie).
*   **Difficulté** : Ajuster le nombre de cases noires pour complexifier ou non la résolution humaine.