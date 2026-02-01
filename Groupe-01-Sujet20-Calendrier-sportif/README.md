# README – Calendrier Sportif Sports Tournament Scheduling

## Présentation du projet
Ce projet a pour objectif de générer automatiquement le calendrier d’un championnat sportif de type round-robin aller-retour. Chaque équipe affronte toutes les autres équipes deux fois : une fois à domicile et une fois à l’extérieur.

Le problème est modélisé comme un problème de programmation par contraintes CSP et résolu à l’aide du solveur CP-SAT de la bibliothèque OR-Tools. L’objectif principal est de produire un calendrier valide et équilibré, tout en minimisant le nombre de breaks matchs consécutifs joués à domicile ou à l’extérieur.

---

## Fonctionnalités principales
- Génération automatique d’un calendrier aller-retour
- Gestion d’un nombre pair ou impair d’équipes (ajout automatique d’une équipe fictive si nécessaire)
- Respect strict des contraintes sportives :
  - Une équipe joue exactement un match par journée
  - Chaque paire d’équipes se rencontre exactement deux fois
  - Alternance domicile / extérieur équilibrée
  - Limitation et minimisation du nombre de breaks
- Optimisation globale du calendrier
- Affichage détaillé des matchs et statistiques dans le terminal
- Visualisation graphique du calendrier via un **diagramme de Gantt**

---

## Technologies utilisées
- **Python 3**
- **OR-Tools CP-SAT Solver**
- **Matplotlib pour la visualisation**

---

## Installation

### Prérequis
- Python 3.9 ou supérieur

### Installation des dépendances
```bash
pip install ortools matplotlib
```

---

## Lancement du programme

Exécuter le fichier principal :
```bash
python main.py
```

Le programme :
1. Initialise le tournoi avec un nombre d’équipes donné
2. Modélise le problème sous forme de CP
3. Résout le problème avec CP-SAT
4. Affiche le calendrier et les statistiques
5. Génère une visualisation graphique

---

## Paramétrage

Le nombre d’équipes est défini dans la fonction `main()` :
```python
scheduler = TournamentScheduler(n_teams=8)
```

Si le nombre d’équipes est impair, le programme rend automatiquement le nombre de d'équipes pair.

---

## Tests et validation

La validité des solutions est assurée par :
- Les contraintes strictes imposées au solveur
- Les statistiques affichées par équipe: domicile, extérieur, breaks
- L’analyse visuelle du diagramme de Gantt

Des tests ont été effectués pour différents nombres d’équipes (4 à 12), avec des solutions systématiquement valides (nombre de break toujours egaux à n-2 avec n le nombre d'équipes).

---

## Références
- Régin, J.-C. (2008). *Minimizing Breaks in Sports Schedules*
- Schaerf, A. (1999). *Sports Scheduling: A Survey*
- International Timetabling Competition (ITC)

---

