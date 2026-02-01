
# Documentation technique – Génération de calendrier sportif

## 1. Problème traité
Le projet traite un problème d’ordonnancement sportif : la génération d’un calendrier round-robin aller-retour pour un championnat.

- n équipes (nombre pair)
- J = 2(n − 1) journées
- Chaque équipe joue exactement un match par journée
- Chaque paire d’équipes se rencontre exactement deux fois domicile / extérieur

L’objectif est de produire un calendrier :
- valide sportivement
- équilibré domicile / extérieur
- minimisant le nombre de breaks

---

## 2. Modélisation par programmation par contraintes
Le problème est modélisé comme un CP (Programmation par Contraintes) avec optimisation.

### 2.1 Variables de décision

#### Matchs
`match_vars[t, i, j] ∈ {0,1}`  
Vaut 1 si l’équipe i reçoit l’équipe j lors de la journée t.

#### Localisation domicile / extérieur
`is_home[t, i] ∈ {0,1}`  
Vaut 1 si l’équipe i joue à domicile à la journée t.

#### Breaks
`break_vars[t, i] ∈ {0,1}`  
Vaut 1 si l’équipe i joue deux fois consécutives au même endroit.

---

## 3. Contraintes

### 3.1 Un match par équipe et par journée
Chaque équipe joue exactement un match par journée :
```
∑ (match_vars[t,i,j] + match_vars[t,j,i]) = 1
```

### 3.2 Unicité des confrontations
Chaque paire d’équipes se rencontre exactement deux fois :
```
∑ match_vars[t,i,j] = 1
∑ match_vars[t,j,i] = 1
```

### 3.3 Cohérence domicile / extérieur
Si une équipe reçoit, l’autre joue à l’extérieur :
```
is_home[t,i] + is_home[t,j] = 1
```

### 3.4 Équilibre domicile / extérieur
Chaque équipe joue exactement n−1 matchs à domicile :
```
∑ is_home[t,i] = n − 1
```

### 3.5 Définition des breaks
Un break est détecté si deux journées consécutives ont le même statut :
```
break_vars[t,i] = 1 ⇔ is_home[t,i] = is_home[t+1,i]
```

---

## 4. Fonction objectif
Minimiser le nombre total de breaks :
```
min ∑ break_vars[t,i]
```
Une borne théorique supérieure est fixée à :
```
n − 2
```

---

## 5. Résolution
- Solveur : OR-Tools CP-SAT
- Optimisation par minimisation
- Garantie de validité et d’optimalité

---

## 6. Analyse des résultats
Après résolution :
- Affichage du calendrier par journée
- Calcul des statistiques par équipe
- Vérification du nombre de breaks

---

## 7. Visualisation
Le calendrier est représenté sous forme de diagramme de Gantt, facilitant :
- la lecture globale
- l’analyse des alternances domicile / extérieur

---

## 8. Limites et améliorations possibles
- Contraintes géographiques
- Disponibilité des stades

---

## 9. Conclusion
Ce projet illustre l’efficacité de la programmation par contraintes pour l’ordonnancement sportif, en produisant des solutions optimales, interprétables et réalistes.
