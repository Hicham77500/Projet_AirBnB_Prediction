# Prédiction des Prix Airbnb

## Description
Projet d'analyse prédictive visant à estimer les prix des logements Airbnb. À partir des caractéristiques des annonces (localisation, type de logement, capacité d'accueil), nous avons développé et comparé plusieurs modèles (régression linéaire simple/multiple, Random Forest) pour prédire le prix optimal d'une location.

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

```bash
git clone https://github.com/votre-username/airbnb-price-prediction.git
cd airbnb-price-prediction
pip install -r requirements.txt
```

## Utilisation

1. **Prétraitement des données**
```bash
python src/data_preprocessing.py
```

2. **Entraînement des modèles**
```bash
python src/model_training.py
```

3. **Exploration via les notebooks**
```bash
jupyter notebook notebooks/
```

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
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
joblib
```

## Résultats

Le modèle Random Forest offre les meilleures performances, suivi par la régression linéaire multiple. Les caractéristiques les plus influentes sont le nombre de personnes accueillies, le nombre de chambres et la localisation.
