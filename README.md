# Prédiction des Prix Airbnb

## Description
Projet d'analyse prédictive visant à estimer les prix des logements Airbnb. À partir des caractéristiques des annonces (localisation, type de logement, capacité d'accueil), nous avons développé et comparé plusieurs modèles (régression linéaire simple/multiple, Random Forest) pour prédire le prix optimal d'une location.

## Prérequis
- Python 3.8 ou supérieur
- Accès aux données Airbnb (voir section "Données")

## Structure du Projet
```
airbnb-price-prediction/
│
├── data/
│   ├── raw/                  # Données brutes d'Airbnb (ignorées par git)
│   ├── processed/            # Données nettoyées
│   └── splits/               # Ensembles d'entraînement et de test (ignorés par git)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb     # Analyse exploratoire
│   ├── 02_data_preprocessing.ipynb   # Prétraitement des données
│   └── 03_model_training.ipynb       # Entraînement des modèles
│
├── models/                   # Modèles entraînés sauvegardés
│
├── results/                  # Visualisations et métriques
│
└── src/                      # Code source Python
    ├── data_preprocessing.py # Nettoyage des données
    ├── feature_engineering.py # Création de features
    ├── model_training.py     # Entraînement des modèles
    └── utils.py              # Fonctions utilitaires
```

## Installation

1. **Cloner le dépôt**
```bash
git clone https://github.com/username/airbnb-price-prediction.git
cd airbnb-price-prediction
```

2. **Créer et activer un environnement virtuel (recommandé)**
```bash
python -m venv venv
source venv/bin/activate  # Sur macOS/Linux
# ou
venv\Scripts\activate     # Sur Windows
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

## Données
Les données brutes doivent être placées dans le dossier `data/raw/`. Vous pouvez :
- Télécharger les données depuis [Inside Airbnb](http://insideairbnb.com/get-the-data.html)
- Obtenir les données depuis notre serveur interne à l'adresse `\\serveur\projets\airbnb\data`
- Contacter l'équipe data pour obtenir un accès aux fichiers

Le jeu de données principal doit être nommé `listings.csv` et placé directement dans le dossier `data/raw/`.

## Utilisation

1. **Prétraitement des données**
```bash
python src/data_preprocessing.py
```
Cette étape nettoie les données brutes et crée les fichiers nécessaires dans `data/processed/`.

2. **Entraînement des modèles**
```bash
python src/model_training.py
```
Cette commande entraîne les différents modèles et sauvegarde les résultats dans le dossier `models/`.

3. **Exploration via les notebooks**
```bash
jupyter notebook notebooks/
```
Les notebooks permettent une analyse plus détaillée des données et des résultats.

4. **Visualisation des résultats**
Après l'exécution des modèles, consultez le dossier `results/` pour accéder aux graphiques et métriques générés.

## Fichiers ignorés par Git
Les données brutes et les ensembles d'entraînement/test sont ignorés en raison de leur taille :
```
data/raw/*
data/split/*
*.pyc
__pycache__/
*.ipynb_checkpoints
models/*
data/processed/*
```

## Requirements
```
pandas==1.3.5
numpy==1.21.6
scikit-learn==1.0.2
matplotlib==3.5.3
seaborn==0.11.2
jupyter==1.0.0
joblib==1.1.0
```

## Résultats

Après comparaison des trois modèles, le Random Forest offre les meilleures performances :

### Performances des modèles (sur l'ensemble de test)
| Modèle | RMSE | R² |
|--------|------|-----|
| Régression linéaire simple (accommodates) | 43.81 | 0.12 |
| Régression linéaire multiple | 46.55 | 0.003 |
| **Random Forest** | **35.70** | **0.41** |

Le modèle Random Forest actuel présente les caractéristiques les plus influentes suivantes :
1. Latitude (importance: 0.17)
2. ID (importance: 0.17) - *Note: ceci indique un problème potentiel*
3. Longitude (importance: 0.16)
4. Nombre de personnes accueillies (importance: 0.12)
5. Nombre d'avis (importance: 0.09)

### Améliorations prévues
- Exclure l'ID des variables prédictives (fuite de données)
- Ajouter le calcul du MAE (Mean Absolute Error)
- Améliorer le prétraitement des données pour augmenter le R²
- Implémenter une validation croisée pour évaluer la stabilité du modèle

*Pour des visualisations détaillées et les résultats complets, consultez le notebook 03_model_training.ipynb ou exécutez `python src/model_training.py`.*
