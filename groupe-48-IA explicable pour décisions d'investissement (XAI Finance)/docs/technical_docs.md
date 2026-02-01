# Documentation Technique - XAI Finance

## 1. Architecture Globale

Le projet suit une architecture découplée **Client-Serveur** :

- **Frontend (Web)** : Application développée avec **Next.js 16** et **React 19**, utilisant **Tailwind CSS v4** pour le style et **Recharts** pour la visualisation de données.
- **Backend (API)** : API REST construite avec **FastAPI** (Python). Elle expose les modèles d'IA, gère le pipeline de données et l'explicabilité.
- **Base de Données** : **Supabase (PostgreSQL)** héberge les données financières (prix, macroéconomie, indicateurs techniques, fondamentaux).

### Schéma de Communication
```mermaid
graph LR
    User[Utilisateur] -->|Requête HTTP| Frontend[Next.js App]
    Frontend -->|API Call /analysis/{ticker}| Backend[FastAPI]
    Backend -->|Fetch Data| DB[Supabase]
    Backend -->|Inference| ML[XGBoost Model]
    ML -->|SHAP Values| XAI[Explainability Module]
    XAI -->|Context| LLM[LLM / Rule-Based]
    LLM -->|Commentary| Backend
    Backend -->|JSON Response| Frontend
```

## 2. Pipeline de Données (Backend)

Le pipeline de données est géré par le module `src/backend/data_loader.py`.

### Sources de Données
Les données sont stockées dans Supabase et divisées en tables :
- `prices_daily` : Historique OHLCV.
- `macro_indicators` : Données macroéconomiques (Taux d'intérêt, inflation, etc.).
- `technical_indicators` : RSI, MACD, etc. calculés en amont.
- `fundamentals_serving` : Ratios financiers (PE Ratio, ROE, etc.).

### Chargement et Mise en Cache
- Le système utilise un mécanisme de **cache local (.parquet)** dans le dossier `data/` pour éviter les appels répétitifs à la base de données lors du développement et de l'entraînement.
- Le fichier `src/backend/preprocessing.py` nettoie et fusionne ces sources en un DataFrame unique synchronisé sur les dates de trading.

## 3. Intelligence Artificielle (ML & XAI)

### Modèle de Prédiction (`src/backend/model.py`)
- **Algorithme** : **XGBoost Classifier**.
- **Objectif** : Classification Binaire (LONG si la probabilité de hausse est > 50%, sinon SHORT).
- **Entraînement** :
    - Utilise une approche **Walk-Forward Validation** (réentraînement mensuel simulé) pour s'adapter aux changements de régime de marché.
    - Paramètres optimisés : `max_depth=3`, `learning_rate=0.01` (pour éviter le surapprentissage).
- **Features** : Combinaison d'indicateurs techniques (RSI, Moyennes Mobiles), de données macro (Taux 10 ans, VIX), et de fondamentaux.

### Explicabilité (XAI) (`src/backend/explainability.py`)
Le projet met l'accent sur la transparence des décisions du modèle.
- **Méthode** : **SHAP (SHapley Additive exPlanations)** via `TreeExplainer`.
- **Fonctionnement** :
    1. Pour chaque prédiction, le système calcule la contribution marginale de chaque feature (ex: RSI, Taux directeurs) au score final.
    2. Les contributions positives sont classées comme "Arguments Haussiers" (Bullish).
    3. Les contributions négatives sont classées comme "Arguments Baissiers" (Bearish).
    4. Ces arguments sont traduits en langage naturel (ex: "RSI is Oversold -> ↑").

### Génération de Commentaire (LLM) (`src/backend/llm_utils.py`)
Une couche de synthèse en langage naturel est ajoutée pour rendre l'analyse accessible.
- **Moteur** : Hybride.
    - Tente d'abord d'utiliser **Claude 3 (Anthropic)** via API.
    - Fallback sur **GPT-3.5 (OpenAI)**.
    - Fallback final sur un **Générateur à base de règles** (si aucune API Key n'est présente).
- **Prompt** : Le système injecte la prédiction, la confiance, et les top arguments SHAP (Bullish/Bearish) pour générer un paragraphe de synthèse financière "professionnelle".

## 4. API Reference

L'API tourne par défaut sur le port `8000`.

### `GET /tickers`
Retourne la liste des tickers disponibles localement pour l'analyse.
- **Réponse** : `["AAPL", "MSFT", ...]`

### `GET /search?q={query}`
Recherche d'actifs via Yahoo Finance (YahooQuery) et filtrage par disponibilité locale.
- **Paramètres** : `q` (str) - Le terme de recherche.
- **Réponse** : Liste d'objets `[{"symbol": "AAPL", "name": "Apple Inc.", ...}]`.

### `GET /analysis/{ticker}`
Lance le pipeline complet d'analyse pour un actif donné.
- **Processus** : Charge les données -> Prétraitement -> Inférence XGBoost -> Calcul SHAP -> Génération Commentaire.
- **Réponse** (JSON) :
```json
{
  "ticker": "AAPL",
  "date": "2024-03-20",
  "prediction": "LONG",
  "confidence": 0.65,
  "base_probability": 0.5,
  "bullish_args": [
    {"feature": "rsi_14", "value": 28.5, "shap": 0.12, "text": "RSI is Oversold..."}
  ],
  "bearish_args": [...],
  "ai_commentary": "The model forecasts a strong Bullish trend...",
  "current_price": 175.50
}
```

## 5. Frontend (Web)

L'application se trouve dans `src/web/`.
- **Pages** :
    - `/` : Tableau de bord principal avec barre de recherche.
    - Affiche les résultats sous forme de cartes (Score IA, Graphique de Prix, Arguments Pour/Contre).
- **Composants Clés** :
    - `AnalysisCard` : Affiche la synthèse de la prédiction.
    - `ArgumentList` : Liste visuelle des facteurs déterminants (expliqués par SHAP).
    - `AICommentary` : Bloc de texte généré par le LLM.

## 6. Installation & Configuration Technique

### Prérequis
- Fichier `.env.local` contenant :
    - `NEXT_PUBLIC_SUPABASE_URL` & `NEXT_PUBLIC_SUPABASE_ANON_KEY` (Accès DB).
    - `ANTHROPIC_API_KEY` ou `OPENAI_API_KEY` (Optionnel pour LLM avancé).
- Python 3.8+ avec les dépendances listées dans `requirements.txt`.
- Node.js 18+ pour le frontend.
