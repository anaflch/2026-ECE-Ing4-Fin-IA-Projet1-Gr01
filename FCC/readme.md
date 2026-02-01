Scoring de crédit équitable par optimisation sous contraintes

1. Contexte et objectif du projet

Les systèmes de scoring de crédit sont aujourd’hui largement automatisés à l’aide de modèles de machine learning.
Cependant, ces modèles peuvent reproduire ou amplifier des biais discriminatoires présents dans les données historiques, en défavorisant certains groupes (par exemple selon le sexe ou la nationalité).

L’objectif de ce projet est de :

Concevoir un système d’intelligence artificielle de scoring de crédit intégrant explicitement des contraintes d’équité, afin de contrôler et quantifier le compromis entre performance prédictive et non-discrimination.

⸻

2. Problématique étudiée

Le projet répond à la question suivante :

Comment intégrer formellement des contraintes d’équité dans un modèle de scoring de crédit, tout en conservant des performances prédictives acceptables ?

Pour cela, le problème est formulé comme une optimisation sous contraintes, où les métriques d’équité (Demographic Parity, Equalized Odds) sont imposées directement lors de l’apprentissage du modèle.

⸻

3. Jeu de données

Le projet utilise un jeu de données clients réaliste (clients.csv) contenant :

🔹 Variable cible
	•	default : défaut de paiement (0 = non, 1 = oui)

🔹 Attribut sensible
	•	sex : utilisé pour mesurer et contraindre l’équité du modèle

🔹 Variables explicatives
	•	Données financières : income, credit_amount, loan_duration
	•	Stabilité professionnelle : employment_years
	•	Situation personnelle : marital_status, housing_status, dependents
	•	Niveau d’éducation : education_level

La colonne name est supprimée lors du pré-traitement car elle ne contient aucune information utile pour la prédiction.

FCC/
├── src/
│   ├── config.py           # Configuration (chemins, colonnes)
│   ├── preprocessing.py    # Pré-traitement des données
│   ├── models.py           # Modèles ML de base
│   ├── fairness.py         # Contraintes d’équité (Fairlearn)
│   ├── evaluate.py         # Métriques de performance et d’équité
│   ├── explain.py          # Explicabilité (SHAP)
│   ├── main.py             # Point d’entrée du projet
│   └── plot_results.py     # Génération des graphiques
├── data/
│   ├── raw/clients.csv
│   └── processed/results.json
└── requirements.txt

5. Approche méthodologique

5.1 Modèle de base (baseline)

Un modèle de régression logistique est entraîné sans contrainte d’équité.

Objectif :
	•	Maximiser la performance prédictive (accuracy, AUC)
	•	Servir de point de comparaison

Ce modèle est performant, mais présente des différences de traitement entre groupes.

⸻

5.2 Mesure de l’équité

Les métriques suivantes sont utilisées :
	•	Demographic Parity Difference (DP)
Différence de taux d’acceptation entre groupes
	•	Equalized Odds Difference (EO)
Différence de faux positifs et faux négatifs entre groupes

Ces métriques permettent de quantifier objectivement la discrimination du modèle.

⸻

5.3 Modèles équitables (in-processing)

L’équité est intégrée directement dans l’apprentissage grâce à la librairie Fairlearn, via l’algorithme :
	•	Exponentiated Gradient Reduction

Deux contraintes sont étudiées :
	•	Demographic Parity
	•	Equalized Odds

Le paramètre epsilon contrôle le niveau de tolérance à la violation de l’équité.

⸻

5.4 Analyse du compromis équité / performance

Le projet fait varier epsilon afin d’observer :
	•	la réduction progressive des biais
	•	l’impact sur la performance prédictive

Cette analyse permet de montrer que l’équité est un choix de gouvernance, et non une propriété binaire.

⸻

6. Résultats principaux

Un fichier results.json est généré automatiquement et contient :
	•	performances (accuracy, AUC)
	•	métriques d’équité (dp_diff, eo_diff)
	•	métriques par groupe

Des graphiques sont produits :
	•	Trade-off AUC vs epsilon
	•	Trade-off Demographic Parity vs epsilon
	•	Taux d’acceptation par groupe (baseline vs modèles équitables)

🔍 Observation clé
	•	Le modèle de base est le plus performant mais le plus discriminant
	•	Les modèles équitables réduisent fortement les biais
	•	La perte de performance reste modérée et contrôlable

⸻
8. Installation et exécution

Création de l’environnement virtuel :
    python3 -m venv .venv
    source .venv/bin/activate

Installation des dépendances :
    pip install -r FCC/requirements.txt

Lancement du projet :
    python -m FCC.src.main

Génération des graphiques :
    python -m FCC.src.plot_results

---

## 👥 Équipe
- **Hugo**
- **Jeremy**
- **Mael**

Projet réalisé dans le cadre du cours *Intelligence Artificielle – Finance*
(ECE Paris, Ingénieur 4ᵉ année).
