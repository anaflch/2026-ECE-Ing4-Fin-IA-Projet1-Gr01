# XAI Finance - IA Explicable pour Décisions d'Investissement

## Description
Projet d'Intelligence Artificielle appliquée à la finance, visant à fournir des analyses boursières explicables (XAI) pour aider à la prise de décision d'investissement. Le projet combine l'analyse de données fondamentales, techniques, et d'actualités via des modèles d'IA et une interface web interactive.

## Structure du Projet

- `src/` : Code source complet.
    - `backend/` : Scripts Python, modèles ML, et API (anciennement à la racine).
    - `web/` : Application Web (Next.js/React).
    - `notebooks/` : Notebooks Jupyter d'exploration et d'entraînement.
- `docs/` : Documentation technique et fonctionnelle.
- `slides/` : Support de présentation.
- `data/` : Données brutes ou traitées (si applicable).

## Installation

### Prérequis
- Python 3.8+
- Node.js 18+
- npm ou yarn

### 1. Backend (Python)
1. Naviguez vers le dossier backend :
   ```bash
   cd src/backend
   ```
2. Installez les dépendances :
   ```bash
   pip install -r ../../requirements.txt
   # Ou si un requirements.txt spécifique existe dans backend :
   # pip install -r requirements.txt
   ```
3. **Configuration des Clés (CRITIQUE)** :
   Le projet nécessite des clés API pour fonctionner (Base de données & IA).
   
   - Un fichier modèle `src/web/.env.example` est fourni.
   - **Pour les correcteurs** : Créez un fichier `.env.local` dans `src/web/` et collez-y les clés fournies dans le commentaire de rendu.
   
   ```bash
   # Exemple de commande
   cp .env.example .env.local
   # Puis éditez le fichier avec vos clés
   ```

### 2. Frontend (Web)
1. Naviguez vers le dossier web :
   ```bash
   cd src/web
   ```
2. Installez les dépendances :
   ```bash
   npm install
   ```
3. Lancez le serveur de développement :
   ```bash
   npm run dev
   ```
   L'application sera accessible sur `http://localhost:3000`.

## Utilisation

### Lancement de l'API / Scripts
- Pour lancer l'entraînement ou les scripts d'analyse, référez-vous aux scripts dans `src/backend`.
- Exemple : `python src/backend/main.py` (à adapter selon le point d'entrée réel).

### Interface Web
- Une fois le serveur Next.js lancé, utilisez l'interface pour visualiser les analyses et les recommandations.

## Tests

### Backend
Pour exécuter les tests unitaires (si disponibles) :
```bash
pytest src/backend/tests
```

### Frontend
Pour exécuter les tests frontend :
```bash
cd src/web
npm test
```

## Documentation
La documentation technique détaillée se trouve dans le dossier `docs/`.

## Auteurs
Groupe 48 - ECE Paris
Alexis Thébault
Maxime Delplace 
Malek Boussofara

