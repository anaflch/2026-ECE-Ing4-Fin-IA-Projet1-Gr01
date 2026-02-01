# Documentation Technique - Scoring de Crédit Équitable

## Table des matières
1. [Vue d'ensemble](#vue-densemble)
2. [Architecture du projet](#architecture-du-projet)
3. [Technologies et dépendances](#technologies-et-dépendances)
4. [Configuration](#configuration)
5. [Modules et composants](#modules-et-composants)
6. [Pipeline de traitement](#pipeline-de-traitement)
7. [Algorithmes et méthodes](#algorithmes-et-méthodes)
8. [Métriques d'évaluation](#métriques-dévaluation)
9. [Visualisation](#visualisation)
10. [Tests](#tests)
11. [Utilisation](#utilisation)

---

## Vue d'ensemble

Le projet implémente un système de scoring de crédit intégrant des contraintes d'équité formelles. Il utilise des techniques d'**apprentissage machine équitable** pour contrôler et quantifier le compromis entre performance prédictive et non-discrimination.

### Objectif principal
Concevoir un modèle de classification qui prédit le défaut de paiement tout en garantissant un traitement équitable par ML sans que ça pose de problèmes d'équité (biais contre certains groupes).

---

## Architecture du projet

```
FCC/
├── src/                            # Code source principal
│   ├── __init__.py                 # Package Python
│   ├── main.py                     # Point d'entrée principal
│   ├── config.py                   # Configuration centralisée
│   ├── preprocessing.py            # Prétraitement des données
│   ├── models.py                   # Modèles de base (baseline)
│   ├── fairness.py                 # Modèles équitables et métriques
│   ├── evaluate.py                 # Évaluation des modèles
│   ├── train.py                    # Script d'entraînement alternatif
│   ├── explain.py                  # Explicabilité (SHAP)
│   ├── plot_results.py             # Génération des graphiques
│   ├── print_results.py            # Affichage des résultats
│   ├── metrics.py                  # (réservé pour métriques custom)
│   ├── utils.py                    # (réservé pour utilitaires)
│   ├── data_loader.py              # (réservé pour chargement données)
│   ├── data/
│   │   ├── raw/
│   │   │   └── clients.csv         # Données brutes
│   │   └── processed/
│   │       ├── results.json        # Résultats d'évaluation
│   │       ├── *.png               # Graphiques générés
│   └── tests/
│       ├── test_fairness_metrics.py
│       └── test_smoke.py
├── notebooks/                      # Notebooks Jupyter
│   ├── 01_baseline.ipynb           # Analyse baseline
│   ├── 02_fair_inprocessing.ipynb  # Modèles équitables
│   ├── 03_postprocessing.ipynb     # Post-traitement
│   └── 04_tradeoff_analysis.ipynb  # Analyse des compromis
├── docs/
│   └── documentation_technique.md  # Documentation technique
├── requirements.txt                # Dépendances Python
└── readme.md                       # Documentation utilisateur
```

---

## Technologies et dépendances

### Bibliothèques Python principales

| Bibliothèque | Version | Rôle |
|--------------|---------|------|
| **numpy** | latest | Calcul numérique et manipulation de tableaux |
| **pandas** | latest | Manipulation de données tabulaires |
| **scikit-learn** | latest | Modèles ML, métriques, prétraitement |
| **fairlearn** | latest | Contraintes d'équité et métriques de fairness |
| **matplotlib** | latest | Visualisation de données |
| **seaborn** | latest | Visualisation statistique avancée |
| **shap** | latest | Explicabilité des modèles (SHAP values) |
| **cvxpy** | latest | Optimisation convexe (utilisé par Fairlearn) |

### Installation

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou .venv\Scripts\activate  # Windows
pip install -r FCC/requirements.txt
```

---

## Configuration

### Module [`config.py`](../src/config.py)

Le fichier [`config.py`](../src/config.py) centralise toute la configuration du projet via des **dataclasses immutables** (`frozen=True`).

#### Classe `Paths`
Gère les chemins de fichiers de manière robuste :
```python
@dataclass(frozen=True)
class Paths:
    ROOT: Path = Path(__file__).resolve().parents[1]
    DATA_RAW: Path = ROOT / "src" / "data" / "raw" / "clients.csv"
    ARTIFACTS: Path = ROOT / "src" / "data" / "processed"
```

- `ROOT` : Répertoire racine du projet FCC
- `DATA_RAW` : Chemin vers les données brutes
- `ARTIFACTS` : Répertoire de sortie pour résultats et artefacts

#### Classe `Columns`
Définit les colonnes clés du dataset :
```python
@dataclass(frozen=True)
class Columns:
    TARGET: str = "default"      # Variable cible (0/1)
    SENSITIVE: str = "sex"       # Attribut sensible pour l'équité
```

#### Classe `Split`
Paramètres pour la division train/test :
```python
@dataclass(frozen=True)
class Split:
    TEST_SIZE: float = 0.3       # 30% des données pour le test
    RANDOM_STATE: int = 42       # Reproductibilité
```

---

## Modules et composants

### 1. [`preprocessing.py`](../src/preprocessing.py) - Prétraitement des données

#### Fonction `load_dataframe(path: str) -> pd.DataFrame`
- Charge le fichier CSV brut
- Retourne un DataFrame pandas

#### Fonction `prepare_splits(...)`
**Pipeline de prétraitement complet** :

```python
def prepare_splits(df, target, sensitive, test_size, random_state):
    # 1. Extraction des variables
    y = df[target].astype(int)           # Variable cible
    A = df[sensitive]                     # Attribut sensible
    X = df.drop(columns=[target])         # Features
    
    # 2. Encodage des variables catégorielles
    X = pd.get_dummies(X, drop_first=True)  # One-hot encoding
    
    # 3. Split train/test stratifié
    X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
        X, y, A, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 4. Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, A_train, A_test, feature_names
```

**Caractéristiques** :
- **One-hot encoding** avec `drop_first=True` pour éviter la multicolinéarité
- **Split stratifié** pour préserver la distribution de la classe cible
- **Standardisation** (moyenne=0, écart-type=1) pour normaliser les features

---

### 2. [`models.py`](../src/models.py) - Modèles de base

#### `train_baseline_logreg(X_train, y_train)`
Entraîne un modèle de **régression logistique** sans contrainte d'équité :
```python
model = LogisticRegression(max_iter=2000, n_jobs=None)
model.fit(X_train, y_train)
```
- `max_iter=2000` : Nombre maximal d'itérations pour la convergence
- Sert de **baseline** pour comparer les performances

#### `train_baseline_rf(X_train, y_train)`
Entraîne un modèle de **Random Forest** :
```python
model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
```
- Alternative plus complexe pour comparaison
- `n_jobs=-1` : Parallélisation sur tous les cœurs disponibles

---

### 3. [`fairness.py`](../src/fairness.py) - Équité et contraintes

Ce module contient les fonctionnalités liées à l'équité algorithmique.

#### Fonction `compute_performance(y_true, y_pred, y_proba)`
Calcule les métriques de **performance prédictive** :
```python
{
    "accuracy": accuracy_score(y_true, y_pred),
    "auc": roc_auc_score(y_true, y_proba)  # si y_proba disponible
}
```

#### Fonction `compute_fairness(y_true, y_pred, sensitive_features)`
Calcule les **métriques d'équité** :
- **MetricFrame** : Table de métriques par groupe
  - `selection_rate` : Taux de prédictions positives par groupe
  - `FPR` : Taux de faux positifs par groupe
  - `FNR` : Taux de faux négatifs par groupe
  
- **Différences globales** :
  - `dp_diff` : **Demographic Parity Difference**
  - `eo_diff` : **Equalized Odds Difference**

```python
mf = MetricFrame(
    metrics={"selection_rate": selection_rate, "FPR": false_positive_rate, ...},
    y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features
)
dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features)
eo_diff = equalized_odds_difference(y_true, y_pred, sensitive_features)
```

#### Fonction `train_fair_model(X_train, y_train, A_train, constraint, eps)`
**Algorithme principal** : Entraînement avec contraintes d'équité.

##### Paramètres :
- `constraint` : Type de contrainte
  - `"dp"` → **Demographic Parity** (parité démographique)
  - `"eo"` → **Equalized Odds** (égalité des chances)
- `eps` : Tolérance de violation (0.0 = équité stricte, plus élevé = plus permissif)

##### Implémentation :
```python
if constraint == "dp":
    moment = DemographicParity()
elif constraint == "eo":
    moment = EqualizedOdds()

base_estimator = LogisticRegression(max_iter=2000)

mitigator = ExponentiatedGradient(
    estimator=base_estimator,
    constraints=moment,
    eps=eps
)

mitigator.fit(X_train, y_train, sensitive_features=A_train)
return mitigator
```

##### Algorithme **Exponentiated Gradient Reduction** :
- Méthode d'apprentissage **in-processing** (contraintes intégrées pendant l'entraînement)
- Formulation comme un **jeu à deux joueurs** :
  1. Le modèle maximise la performance
  2. Un adversaire pénalise les violations d'équité
- Converge vers un équilibre où la contrainte d'équité est respectée à `eps` près

---

### 4. [`evaluate.py`](../src/evaluate.py) - Évaluation

#### Fonction `predict_any(model, X)`
Gère les prédictions pour tous types de modèles :
```python
y_pred = model.predict(X)
y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
```
- Compatible avec les modèles Fairlearn et scikit-learn

#### Fonction `evaluate_model(name, model, X_test, y_test, A_test)`
**Pipeline d'évaluation complet** :
1. Génère les prédictions
2. Calcule les métriques de performance
3. Calcule les métriques d'équité
4. Retourne un dictionnaire structuré :

```python
{
    "name": "baseline_logreg",
    "accuracy": 0.85,
    "auc": 0.90,
    "dp_diff": 0.15,
    "eo_diff": 0.12,
    "by_group": {
        "selection_rate": {"Female": 0.60, "Male": 0.75},
        "FPR": {...},
        "FNR": {...}
    },
    "overall": {...}
}
```

---

### 5. [`main.py`](../src/main.py) - Pipeline principal

**Point d'entrée** du projet qui orchestre l'ensemble du workflow.

#### Workflow complet :

```python
def main():
    # 1. Initialisation
    Paths.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    
    # 2. Chargement et préparation des données
    df = load_dataframe(str(Paths.DATA_RAW))
    X_train, X_test, y_train, y_test, A_train, A_test, feature_names = prepare_splits(...)
    
    # 3. Entraînement du modèle baseline
    base = train_baseline_logreg(X_train, y_train)
    res_base = evaluate_model("baseline_logreg", base, X_test, y_test, A_test)
    
    # 4. Entraînement des modèles équitables
    fair_dp = train_fair_model(X_train, y_train, A_train, constraint="dp", eps=0.02)
    res_dp = evaluate_model("fair_dp_eps0.02", fair_dp, X_test, y_test, A_test)
    
    fair_eo = train_fair_model(X_train, y_train, A_train, constraint="eo", eps=0.02)
    res_eo = evaluate_model("fair_eo_eps0.02", fair_eo, X_test, y_test, A_test)
    
    # 5. Analyse du compromis (trade-off sweep)
    eps_grid = [0.005, 0.01, 0.02, 0.05, 0.1]
    sweep = []
    for eps in eps_grid:
        m = train_fair_model(X_train, y_train, A_train, constraint="dp", eps=eps)
        sweep.append(evaluate_model(f"fair_dp_eps{eps}", m, X_test, y_test, A_test))
    
    # 6. Sauvegarde des résultats
    out = {"baseline": res_base, "fair_dp": res_dp, "fair_eo": res_eo, "sweep_dp": sweep}
    out_path = Paths.ARTIFACTS / "results.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
```

**Résultat** : Fichier [`results.json`](../src/data/processed/results.json) contenant toutes les métriques.

---

### 6. [`plot_results.py`](../src/plot_results.py) - Visualisation

Génère trois graphiques à partir de [`results.json`](../src/data/processed/results.json) :

#### 1. **Trade-off AUC vs epsilon**
```python
plt.plot(eps_vals, auc_vals, marker="o")
plt.xlabel("epsilon (contrainte DP)")
plt.ylabel("AUC")
plt.title("Trade-off performance : AUC vs epsilon")
```
- Montre comment la performance diminue quand on renforce l'équité (eps → 0)

#### 2. **Trade-off dp_diff vs epsilon**
```python
plt.plot(eps_vals, dp_vals, marker="o")
plt.xlabel("epsilon (contrainte DP)")
plt.ylabel("Demographic parity difference (dp_diff)")
```
- Montre comment la discrimination diminue quand on renforce l'équité

#### 3. **Taux d'acceptation par groupe**
```python
plt.bar([i - width for i in x], base_vals, width=width, label="baseline")
plt.bar([i for i in x], dp_vals_b, width=width, label="fair DP")
plt.bar([i + width for i in x], eo_vals_b, width=width, label="fair EO")
```
- Compare le taux de prédictions positives entre groupes pour chaque modèle
- Met en évidence les disparités de traitement

**Sauvegarde** : Fichiers PNG dans `FCC/src/data/processed/`

---

### 7. [`explain.py`](../src/explain.py) - Explicabilité

#### Fonction `shap_explain_logreg(model, X_background, X_explain, feature_names)`
Utilise **SHAP (SHapley Additive exPlanations)** pour l'interprétabilité :

```python
explainer = shap.LinearExplainer(model, X_background, feature_perturbation="interventional")
shap_values = explainer.shap_values(X_explain)
```

**SHAP** attribue une importance à chaque feature pour chaque prédiction :
- Basé sur la théorie des jeux (valeurs de Shapley)
- `LinearExplainer` : optimisé pour les modèles linéaires
- `feature_perturbation="interventional"` : modélise les dépendances entre features

---

### 8. [`print_results.py`](../src/print_results.py) - Affichage

Utilitaire simple pour afficher les résultats en CSV :
```python
print("MODEL,accuracy,auc,dp_diff,eo_diff")
for k in ["baseline", "fair_dp", "fair_eo"]:
    r = d[k]
    print(f"{r['name']},{r.get('accuracy','')},{r.get('auc','')},...")
```

---

## Pipeline de traitement

### Diagramme de flux

```
┌─────────────────┐
│  clients.csv    │
│  (données brutes)│
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  preprocessing.py       │
│  • Chargement           │
│  • One-hot encoding     │
│  • Split train/test     │
│  • Standardisation      │
└────────┬────────────────┘
         │
         ├──────────┬─────────────┬────────────┐
         ▼          ▼             ▼            ▼
    ┌─────────┐ ┌────────┐  ┌──────────┐ ┌──────────┐
    │Baseline │ │Fair DP │  │ Fair EO  │ │Trade-off │
    │LogReg   │ │eps=0.02│  │ eps=0.02 │ │  Sweep   │
    └────┬────┘ └───┬────┘  └────┬─────┘ └────┬─────┘
         │          │            │            │
         └──────────┴────────────┴────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  evaluate.py    │
                  │  • Performance  │
                  │  • Équité       │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  results.json   │
                  └────────┬────────┘
                           │
                  ┌────────┴────────┐
                  ▼                 ▼
          ┌──────────────┐  ┌──────────────┐
          │plot_results  │  │print_results │
          │  (PNG)       │  │  (CSV)       │
          └──────────────┘  └──────────────┘
```

---

## Algorithmes et méthodes

### 1. Modèle baseline : Régression logistique

**Formulation mathématique** :
$$P(y=1|x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$$

**Optimisation** (Maximum de vraisemblance) :
$$\min_{w,b} -\sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]$$

**Avantages** :
- Interprétable (coefficients = impact de chaque feature)
- Rapide à entraîner
- Probabiliste (permet le calibrage)

---

### 2. Exponentiated Gradient Reduction (Fairlearn)

#### Principe
Formule l'apprentissage équitable comme un **problème d'optimisation sous contraintes** :

$$
\begin{align}
\min_{h \in \mathcal{H}} & \quad \mathbb{E}[\ell(h(X), Y)] \\
\text{s.t.} & \quad \max_{a \in \mathcal{A}} |P(\hat{Y}=1|A=a) - P(\hat{Y}=1)| \leq \epsilon
\end{align}
$$

Où :
- $\ell$ : fonction de perte (erreur de classification)
- $\mathcal{H}$ : espace des hypothèses (modèles)
- $A$ : attribut sensible
- $\epsilon$ : tolérance de violation

#### Algorithme
1. **Initialisation** : entraîner un modèle $h_0$ sans contrainte
2. **Itération** : pour $t = 1, 2, ..., T$ :
   - Calculer les violations de contraintes pour chaque groupe
   - Ajuster les poids des exemples (upweight les groupes pénalisés)
   - Ré-entraîner un modèle sur les données pondérées
   - Combiner les modèles via une pondération exponentielle
3. **Convergence** : quand les contraintes sont satisfaites à $\epsilon$ près

#### Contraintes supportées

**Demographic Parity (DP)** :
$$P(\hat{Y}=1|A=a) \approx P(\hat{Y}=1|A=b) \quad \forall a,b$$
- Les groupes doivent avoir des taux de prédictions positives similaires
- Ignore la vérité terrain (équité "statistique")

**Equalized Odds (EO)** :
$$P(\hat{Y}=1|Y=y, A=a) \approx P(\hat{Y}=1|Y=y, A=b) \quad \forall y, a, b$$
- Les taux de vrais positifs et faux positifs doivent être égaux entre groupes
- Plus stricte que DP, conditionnée sur la vraie étiquette

---

### 3. SHAP (SHapley Additive exPlanations)

**Valeur de Shapley** pour une feature $j$ :
$$\phi_j = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!(|F|-|S|-1)!}{|F|!} [f(S \cup \{j\}) - f(S)]$$

Où :
- $F$ : ensemble des features
- $S$ : sous-ensemble de features
- $f(S)$ : prédiction du modèle en utilisant seulement les features de $S$

**Interprétation** : contribution marginale moyenne de la feature $j$ sur toutes les coalitions possibles.

---

## Métriques d'évaluation

### Métriques de performance

| Métrique | Formule | Interprétation |
|----------|---------|----------------|
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Proportion de prédictions correctes |
| **AUC-ROC** | $\int_0^1 TPR(FPR^{-1}(x)) dx$ | Aire sous la courbe ROC (capacité de discrimination) |

### Métriques d'équité

#### 1. **Demographic Parity Difference**
$$\text{DP}_{\text{diff}} = \max_{a,b} |P(\hat{Y}=1|A=a) - P(\hat{Y}=1|A=b)|$$

**Interprétation** : Différence maximale de taux d'acceptation entre groupes.
- 0 = équité parfaite
- > 0.1 = discrimination significative

#### 2. **Equalized Odds Difference**
$$\text{EO}_{\text{diff}} = \max_{y \in \{0,1\}, a,b} |P(\hat{Y}=1|Y=y, A=a) - P(\hat{Y}=1|Y=y, A=b)|$$

**Interprétation** : Différence maximale de FPR/TPR entre groupes.
- Plus stricte que DP
- Garantit l'équité conditionnelle

#### 3. **Selection Rate** (par groupe)
$$\text{SR}_a = P(\hat{Y}=1|A=a)$$
Proportion de prédictions positives pour le groupe $a$.

#### 4. **False Positive Rate** (par groupe)
$$\text{FPR}_a = P(\hat{Y}=1|Y=0, A=a)$$

#### 5. **False Negative Rate** (par groupe)
$$\text{FNR}_a = P(\hat{Y}=0|Y=1, A=a)$$

---

## Visualisation

### Graphiques générés

1. **`tradeoff_auc_vs_eps.png`**
   - Axe X : valeur d'epsilon (tolérance de violation)
   - Axe Y : AUC (performance)
   - Observation : L'AUC diminue légèrement quand epsilon diminue (équité plus stricte)

2. **`tradeoff_dp_vs_eps.png`**
   - Axe X : valeur d'epsilon
   - Axe Y : Demographic Parity Difference
   - Observation : dp_diff diminue fortement quand epsilon diminue

3. **`selection_rate_by_group.png`**
   - Barres groupées : baseline / fair DP / fair EO
   - Compare les taux d'acceptation entre groupes pour chaque modèle
   - Révèle les disparités et leur correction

---

## Tests

### [`test_fairness_metrics.py`](../src/tests/test_fairness_metrics.py)
Tests unitaires pour les métriques d'équité (non implémenté actuellement).

### [`test_smoke.py`](../src/tests/test_smoke.py)
Test de non-régression basique (non implémenté actuellement).

**Recommandations** :
```python
# Exemple de test unitaire à implémenter
def test_demographic_parity():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    A = np.array(['a', 'b', 'a', 'b'])
    
    dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=A)
    assert dp_diff == 0.0  # Équité parfaite
```

---

## Utilisation

### 1. Exécution du pipeline complet
```bash
cd FCC
python -m src.main
```

**Résultat** : Génère [`src/data/processed/results.json`](../src/data/processed/results.json)

### 2. Génération des graphiques
```bash
python -m src.plot_results
```

**Résultat** : Génère les PNG dans `src/data/processed/`

### 3. Affichage des résultats
```bash
python -m src.print_results
```

**Résultat** : Affiche un résumé CSV des métriques

### 4. Notebooks Jupyter
```bash
jupyter notebook FCC/notebooks/
```

- [`01_baseline.ipynb`](../notebooks/01_baseline.ipynb) : Exploration et baseline
- [`02_fair_inprocessing.ipynb`](../notebooks/02_fair_inprocessing.ipynb) : Modèles équitables
- [`03_postprocessing.ipynb`](../notebooks/03_postprocessing.ipynb) : Post-traitement
- [`04_tradeoff_analysis.ipynb`](../notebooks/04_tradeoff_analysis.ipynb) : Analyse détaillée des compromis

---

## Considérations techniques

### Performance et scalabilité
- **Temps d'entraînement** : ~5-10 secondes pour le pipeline complet sur CPU moderne
- **Mémoire** : < 500 MB (dataset de taille modérée)
- **Parallélisation** : Possible via `n_jobs=-1` (scikit-learn) et `joblib`

### Reproductibilité
- `random_state=42` : Fixé dans toute la chaîne (split, modèles)
- `numpy.random.seed()` : À définir pour une reproductibilité totale

### Limitations actuelles
1. **Dataset unique** : Conçu pour `clients.csv`
2. **Attribut sensible binaire** : Extensible à plusieurs groupes
3. **Métriques d'équité** : DP et EO uniquement (autres possibles avec Fairlearn)
4. **Tests** : Couverture insuffisante

### Extensions possibles
1. **Post-processing** : Ajuster les seuils de décision par groupe
2. **Autres contraintes** : Equalizing Error Rates, Predictive Parity
3. **Modèles complexes** : Étendre à XGBoost, réseaux de neurones
4. **Multi-attributs** : Intersectionnalité (sexe × âge)
5. **Calibration** : Assurer $P(\hat{Y}=p|A=a) = p$ (calibration par groupe)

---

## Références

### Bibliothèque Fairlearn
- Documentation : https://fairlearn.org/
- Paper : *Fairlearn: A toolkit for assessing and improving fairness in AI* (Microsoft Research, 2020)

### Algorithmes
- **Exponentiated Gradient Reduction** : Agarwal et al. (2018), *A Reductions Approach to Fair Classification*
- **SHAP** : Lundberg & Lee (2017), *A Unified Approach to Interpreting Model Predictions*

### Métriques d'équité
- Hardt et al. (2016), *Equality of Opportunity in Supervised Learning*
- Dwork et al. (2012), *Fairness Through Awareness*

---

## Auteurs et licence

**Équipe** :
- Hugo
- Jeremy
- Mael

**Projet** : Intelligence Artificielle – Finance (ECE Paris, Ing4)

**Licence** : Voir fichier [`LICENSE`](../../LICENSE)

---

## Changelog

- **v1.0** (2026-01) : Version initiale avec baseline, fair models, et trade-off analysis
