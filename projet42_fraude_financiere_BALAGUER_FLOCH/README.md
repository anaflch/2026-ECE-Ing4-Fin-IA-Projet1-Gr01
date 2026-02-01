### Projet 1 : Intelligence Artificielle finance - Jules BALAGUER, Anaïs FLOCH

## Détection de fraude financière par Intelligence Artificielle Symbolique (ILP – Aleph, sujet 42)

## Objectif du projet

Ce projet a pour objectif de détecter des transactions frauduleuses (blanchiment d’argent)
en utilisant une approche **symbolique et explicable** d’Intelligence Artificielle :
l’**Inductive Logic Programming (ILP)**, via l’algorithme **Aleph**.

Contrairement aux approches purement statistiques ou neuronales,
l’ILP permet d’apprendre des **règles logiques interprétables**
à partir de données structurées et de connaissances de domaine.

---
## Important avant de commencer

Les datasets utilisés sont trop volumineux pour être déposés ici. Veuillez vous rendre
sur le lien suivant :
https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
et téléchargez "HI-Small_Trans.csv" et "LI-Small_Trans.csv". Pensez à les placer dans le même
dossier que le reste du projet.
Vous êtes maintenant prêt à démarrer !

---

## Principe général

Le pipeline du projet est le suivant :

1. Chargement et équilibrage d’un dataset de transactions financières
2. Construction d’un **graphe de transactions** (comptes = nœuds, transactions = arêtes)
3. Extraction de **features symboliques** issues du graphe
4. Génération automatique d’une base de connaissances **Prolog**
5. Apprentissage de règles de fraude avec **Aleph**
6. Évaluation des règles apprises sur un jeu de test

L’ensemble du processus est **entièrement automatisé** et ne nécessite aucune interaction manuelle avec Prolog.

---

## Variables modifiables

Il est possible de jouer sur certaines variables :

1. Taille de l'échantillon à analyser : **sample_size** ligne 441
2. Ratio de fraudes dans l'échantillon : **fraud_ratio** ligne 441, en sachant qu'il y a
un maximum de 8742 fraudes dans le dataset

---

## Structure du projet

```text
projet/
├── fraud_detector.py          # Script principal (pipeline complet)
├── LI-Small_Trans.csv         # Dataset (low intensity)
├── HI-Small_Trans.csv         # Dataset (high intensity)
├── fraud_detection.pl         # Généré automatiquement (ILP / Aleph)
├── run_aleph.pl               # Script Prolog pour lancer Aleph
├── learned_rules.pl           # Règles apprises par Aleph
├── test_set.csv               # Jeu de test
├── swipl/                     # SWI-Prolog portable (aucune installation requise)
│   └── bin/swipl.exe
├── aleph/                     # Aleph : IA récupérée sur github
│   └── prolog/aleph.pl
└── README.md
```

---

## Méthodologie (vue d’ensemble)

Le projet repose sur une approche hybride combinant graphes et intelligence artificielle symbolique :

1. Les transactions sont modélisées sous forme de graphe (comptes = nœuds, transactions = arêtes)
2. Des caractéristiques symboliques explicables sont extraites à partir du graphe
3. Ces informations sont traduites en faits et règles Prolog
4. L’algorithme Aleph (ILP) apprend automatiquement des règles de fraude
5. Les règles apprises sont évaluées sur un jeu de test

L’ensemble du pipeline est automatisé et reproductible.

