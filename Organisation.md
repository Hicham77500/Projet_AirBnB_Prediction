## Répartition des tâches pour le projet "IA - AirBnB"

Ce projet vise à préparer les données Airbnb de Paris pour une IA capable de prédire les prix des logements. Il est divisé en trois grandes parties, chacune attribuée à une personne avec des tâches précises et des livrables associés. Voici la répartition :

1. **Préparation des données** (Personne 1)
2. **Exploration et ingénierie des features** (Personne 2)
3. **Modélisation et évaluation** (Personne 3)

---

### Personne 1 : Préparation des données

#### Responsabilités
Vous êtes chargé(e) de charger et nettoyer les données brutes d'Airbnb pour les rendre exploitables par les étapes suivantes du projet. Cela inclut la gestion des valeurs manquantes, la suppression des doublons, et la conversion des types de données.

#### Tâches réalisées
1. **Collecte des données** :
   - Chargement du fichier brut `paris_listings.csv` (situé dans `data/raw/`) avec la bibliothèque `pandas`.
2. **Nettoyage des données** :
   - Sélection des colonnes utiles comme `id`, `price`, `neighbourhood`, `room_type`, `minimum_nights`.
   - Suppression des doublons en utilisant la colonne `id`.
   - Traitement des valeurs manquantes : remplacement par la médiane pour les colonnes numériques (ex. `price`) et par le mode pour les colonnes catégorielles (ex. `room_type`).
   - Conversion des types de données (ex. `price` en `float`, `room_type` en `category`).
3. **Sauvegarde** :
   - Enregistrement des données nettoyées dans `data/processed/paris_listings_cleaned.csv`.

#### Livrables
- Fichier nettoyé : `data/processed/paris_listings_cleaned.csv`.
- Script Python : `src/data_preprocessing.py` contenant les fonctions de nettoyage.

#### Ce que vous pouvez présenter
- Expliquez pourquoi la préparation des données est une étape essentielle dans un projet d'IA.
- Montrez comment vous avez chargé les données et les étapes de nettoyage (ex. code pour gérer les valeurs manquantes).
- Comparez un extrait des données avant et après votre travail pour montrer l'amélioration (ex. tableau ou graphique).
- Détaillez l'impact de votre contribution sur la qualité des données pour les étapes suivantes.

---

### Personne 2 : Exploration et ingénierie des features

#### Responsabilités
Votre rôle est d'explorer les données pour en comprendre les caractéristiques, de préparer les features pour l'IA (encodage, gestion des outliers), et de diviser les données en ensembles d'entraînement et de test.

#### Tâches réalisées
1. **Analyse exploratoire** :
   - Création d'un notebook Jupyter (`notebooks/01_data_exploration.ipynb`) avec des statistiques (ex. moyenne des prix) et des visualisations (ex. histogramme des prix, répartition par quartier).
2. **Ingénierie des features** :
   - Encodage des variables catégorielles comme `room_type` et `neighbourhood` avec la méthode one-hot encoding.
3. **Gestion des outliers** :
   - Suppression des valeurs aberrantes dans `price` en utilisant la méthode IQR (Interquartile Range).
4. **Division des données** :
   - Séparation en ensembles d'entraînement (80%) et de test (20%), sauvegardés dans `data/splits/train.csv` et `data/splits/test.csv`.

#### Livrables
- Notebook : `notebooks/01_data_exploration.ipynb` avec l'analyse et les visualisations.
- Script : `src/feature_engineering.py` pour l'encodage et la préparation des features.
- Fichiers : `data/splits/train.csv` et `data/splits/test.csv`.

#### Ce que vous pouvez présenter
- Présentez vos découvertes (ex. "Les prix varient fortement selon les quartiers" avec un graphique).
- Expliquez comment et pourquoi vous avez encodé les variables catégorielles (ex. one-hot encoding).
- Montrez comment vous avez identifié et supprimé les outliers dans `price`, avec un avant/après.
- Décrivez l'importance de la division des données pour éviter le surapprentissage et tester les modèles.

---

### Personne 3 : Modélisation et évaluation

#### Responsabilités
Vous êtes responsable de construire et d'évaluer des modèles d'IA pour prédire les prix des logements, puis de sauvegarder les résultats pour une utilisation future.

#### Tâches réalisées
1. **Entraînement des modèles** :
   - Création d'une régression linéaire simple avec une seule feature (ex. `minimum_nights`).
   - Création d'une régression linéaire multiple utilisant toutes les features préparées par la Personne 2.
2. **Évaluation** :
   - Calcul des métriques de performance : MSE (Mean Squared Error) et R² pour les deux modèles.
3. **Sauvegarde** :
   - Enregistrement des modèles dans `models/linear_regression.pkl` et `models/multiple_regression.pkl`.

#### Livrables
- Script : `src/model_training.py` avec les fonctions d'entraînement et d'évaluation.
- Modèles : `models/linear_regression.pkl` et `models/multiple_regression.pkl`.
- Résultats : fichier `docs/results.md` avec les métriques MSE et R².

#### Ce que vous pouvez présenter
- Expliquez pourquoi vous avez choisi des régressions linéaires et leur pertinence pour prédire les prix.
- Présentez les résultats (ex. "La régression multiple a un R² de 0,65, mieux que la simple à 0,45").
- Montrez comment vous avez sauvegardé les modèles et pourquoi cela permet de les réutiliser.
- Proposez des idées pour améliorer les performances (ex. tester un modèle Random Forest).

---

## Instructions pour tous

- **Documentation** : Ajoutez une section dans le `README.md` pour décrire votre contribution.