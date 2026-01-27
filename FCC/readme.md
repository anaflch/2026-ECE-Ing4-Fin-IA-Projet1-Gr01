# Scoring de crédit équitable par optimisation sous contraintes

## 📌 Contexte et objectif
Les systèmes de scoring de crédit basés sur des modèles statistiques ou de machine learning
sont largement utilisés pour automatiser les décisions d’octroi de crédit.
Cependant, ces systèmes peuvent produire des décisions biaisées à l’encontre de certains groupes
(même lorsque les variables sensibles ne sont pas explicitement utilisées).

L’objectif de ce projet est d’étudier ces biais et de proposer des méthodes permettant
d’intégrer des contraintes d’équité directement dans le processus de décision,
en formulant le problème comme une optimisation sous contraintes.

---

## 🎯 Problématique
À partir d’un score de risque de crédit, une règle de décision simple (par exemple un seuil global)
peut engendrer des disparités importantes entre groupes protégés.

Nous cherchons à répondre aux questions suivantes :
- Comment mesurer l’inéquité dans une décision automatique ?
- Peut-on réduire ces inégalités sans dégrader excessivement la performance globale ?
- Quel est le coût mesurable de l’équité dans un système de décision financière ?

---

## 🧠 Méthodologie

Le projet est structuré en quatre étapes principales :

### 1️⃣ Baseline naïve
- Construction d’un score de risque de crédit interprétable à partir de variables financières.
- Application d’un seuil global unique pour décider de l’acceptation ou du refus d’un crédit.
- Observation d’un biais massif via la parité démographique (Demographic Parity).

📓 Notebook : `01_baseline.ipynb`

---

### 2️⃣ In-processing équitable (optimisation sous contrainte)
- Conservation du score de risque initial.
- Optimisation de la politique de décision sous une contrainte d’équité :
  
  \[
  |P(\text{accepté} \mid sexe = 0) - P(\text{accepté} \mid sexe = 1)| \le \varepsilon
  \]

- Utilisation de seuils différenciés par groupe pour rendre le problème faisable.
- Analyse de l’impact de la contrainte sur la performance globale.

📓 Notebook : `02_fair_inprocessing.ipynb`

---

### 3️⃣ Post-processing équitable
- Correction a posteriori des décisions sans modifier le score.
- Ajustement naïf des seuils par groupe afin de réduire les disparités.
- Comparaison avec l’approche d’in-processing.

📓 Notebook : `03_postprocessing.ipynb`

---

### 4️⃣ Analyse du compromis équité / performance
- Étude de l’impact du paramètre de tolérance à l’inéquité (ε).
- Mise en évidence d’un compromis non linéaire entre équité et performance.
- Identification d’une zone optimale où une légère tolérance permet
  de conserver une performance élevée tout en réduisant fortement les biais.

📓 Notebook : `04_tradeoff_analysis.ipynb`

---

## 📊 Résultats clés

- La baseline produit un biais important entre les groupes.
- Une contrainte d’équité stricte peut fortement réduire la performance globale.
- Un léger relâchement de la contrainte permet d’atteindre une performance proche de l’optimum,
  tout en limitant fortement les disparités.
- L’in-processing offre un meilleur compromis équité / performance que le post-processing.

Ces résultats montrent que l’équité a un coût mesurable,
mais qu’une intégration intelligente des contraintes permet de limiter ce coût.

---

## 📁 Structure du dépôt

FCC/
├── data/
│ └── raw/ # Données synthétiques de clients
├── notebooks/
│ ├── 01_baseline.ipynb
│ ├── 02_fair_inprocessing.ipynb
│ ├── 03_postprocessing.ipynb
│ └── 04_tradeoff_analysis.ipynb
├── src/ # Fonctions utilitaires (chargement, métriques)
├── requirements.txt
└── README.md


---

## 👥 Équipe
- **Hugo**
- **Jeremy**
- **Mael**

Projet réalisé dans le cadre du cours *Intelligence Artificielle – Finance*
(ECE Paris, Ingénieur 4ᵉ année).
