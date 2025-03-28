"""
Module pour l'entraînement et l'évaluation des modèles de prédiction des prix Airbnb.

Ce module contient des fonctions pour :
- Charger les données d'entraînement et de test.
- Entraîner des modèles de régression linéaire simple, multiple et Random Forest.
- Visualiser les prédictions et les corrélations entre les features.
- Comparer les performances des modèles.

Fonctions principales :
    - load_data : Charge les données d'entraînement et de test.
    - train_simple_linear_regression : Entraîne une régression linéaire simple.
    - train_multiple_linear_regression : Entraîne une régression linéaire multiple.
    - train_random_forest : Entraîne un modèle Random Forest.
    - visualize_predictions : Visualise les prédictions vs les valeurs réelles.
    - plot_correlation_matrix : Affiche la matrice de corrélation des features.
    - main : Point d'entrée principal pour exécuter le pipeline complet.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def load_data(train_path, test_path):
    """
    Charge les données d'entraînement et de test à partir de fichiers CSV.

    Args:
        train_path (str): Chemin vers le fichier CSV des données d'entraînement.
        test_path (str): Chemin vers le fichier CSV des données de test.

    Returns:
        tuple: (X_train, y_train, X_test, y_test) où :
            - X_train, X_test : Features d'entraînement et de test.
            - y_train, y_test : Cibles d'entraînement et de test.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Séparer les features et la cible
    X_train = train_df.drop(columns=['price'])
    y_train = train_df['price']
    X_test = test_df.drop(columns=['price'])
    y_test = test_df['price']

    print(f"Données chargées - X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Données chargées - X_test: {X_test.shape}, y_test: {y_test.shape}")

    return X_train, y_train, X_test, y_test

def train_simple_linear_regression(X_train, X_test, y_train, y_test, feature):
    """
    Entraîne une régression linéaire simple avec une seule feature.

    Args:
        X_train (pd.DataFrame): Features d'entraînement.
        X_test (pd.DataFrame): Features de test.
        y_train (pd.Series): Cible d'entraînement.
        y_test (pd.Series): Cible de test.
        feature (str): Nom de la feature à utiliser pour l'entraînement.

    Returns:
        tuple: (model, metrics, y_test_pred) où :
            - model : Modèle entraîné.
            - metrics : Dictionnaire des métriques de performance.
            - y_test_pred : Prédictions sur les données de test.
    """
    # Vérifier que la feature existe
    if feature not in X_train.columns:
        raise ValueError(f"La feature '{feature}' n'existe pas dans les données d'entraînement")

    model = LinearRegression()
    model.fit(X_train[[feature]], y_train)

    # Prédictions
    y_train_pred = model.predict(X_train[[feature]])
    y_test_pred = model.predict(X_test[[feature]])

    # Calcul des métriques
    metrics = {
        'train_mse': mean_squared_error(y_train, y_train_pred),
        'test_mse': mean_squared_error(y_test, y_test_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred)
    }

    # Affichage des métriques
    print(f"MSE (entraînement): {metrics['train_mse']:.2f}")
    print(f"MSE (test): {metrics['test_mse']:.2f}")
    print(f"RMSE (entraînement): {metrics['train_rmse']:.2f}")
    print(f"RMSE (test): {metrics['test_rmse']:.2f}")
    print(f"R² (entraînement): {metrics['train_r2']:.2f}")
    print(f"R² (test): {metrics['test_r2']:.2f}")

    # Sauvegarde du modèle
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/linear_regression_simple_{feature}.pkl")
    print(f"Modèle sauvegardé dans 'models/linear_regression_simple_{feature}.pkl'")

    return model, metrics, y_test_pred

def train_multiple_linear_regression(X_train, X_test, y_train, y_test):
    """Entraîne une régression linéaire multiple avec toutes les features."""
    print("\nEntraînement de la régression linéaire multiple")
    
    # Sélectionner uniquement les colonnes numériques
    numeric_columns = X_train.select_dtypes(include=['int64', 'float64']).columns
    print(f"Utilisation de {len(numeric_columns)} colonnes numériques sur {X_train.shape[1]} colonnes totales")
    
    if len(numeric_columns) == 0:
        raise ValueError("Aucune colonne numérique disponible pour la régression")
    
    # Utiliser seulement les colonnes numériques
    X_train_numeric = X_train[numeric_columns]
    X_test_numeric = X_test[numeric_columns]
    
    # Vérifier s'il y a des valeurs NaN
    if X_train_numeric.isna().any().any() or y_train.isna().any():
        print("ATTENTION: Valeurs NaN détectées dans les données d'entraînement. Remplacement par 0.")
        X_train_numeric = X_train_numeric.fillna(0)
        y_train = y_train.fillna(0)
    
    if X_test_numeric.isna().any().any() or y_test.isna().any():
        print("ATTENTION: Valeurs NaN détectées dans les données de test. Remplacement par 0.")
        X_test_numeric = X_test_numeric.fillna(0)
        y_test = y_test.fillna(0)
    
    model = LinearRegression()
    model.fit(X_train_numeric, y_train)
    
    # Prédictions
    y_train_pred = model.predict(X_train_numeric)
    y_test_pred = model.predict(X_test_numeric)
    
    # Calcul des métriques
    metrics = {
        'train_mse': mean_squared_error(y_train, y_train_pred),
        'test_mse': mean_squared_error(y_test, y_test_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred)
    }
    
    # Affichage des métriques
    print(f"MSE (entraînement): {metrics['train_mse']:.2f}")
    print(f"MSE (test): {metrics['test_mse']:.2f}")
    print(f"RMSE (entraînement): {metrics['train_rmse']:.2f}")
    print(f"RMSE (test): {metrics['test_rmse']:.2f}")
    print(f"R² (entraînement): {metrics['train_r2']:.2f}")
    print(f"R² (test): {metrics['test_r2']:.2f}")
    
    # Coefficients
    print("\nCoefficients du modèle:")
    coef_df = pd.DataFrame({
        'Feature': numeric_columns,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', ascending=False)
    print(coef_df)
    
    # Sauvegarde du modèle
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/multiple_regression.pkl")
    print("Modèle sauvegardé dans 'models/multiple_regression.pkl'")
    
    return model, metrics, y_test_pred

def train_random_forest(X_train, X_test, y_train, y_test, n_estimators=100, random_state=42):
    """Entraîne un modèle Random Forest."""
    print("\nEntraînement d'un modèle Random Forest...")
    
    # Sélectionner uniquement les colonnes numériques
    numeric_columns = X_train.select_dtypes(include=['int64', 'float64']).columns
    print(f"Utilisation de {len(numeric_columns)} colonnes numériques sur {X_train.shape[1]} colonnes totales")
    
    # Utiliser seulement les colonnes numériques
    X_train_numeric = X_train[numeric_columns]
    X_test_numeric = X_test[numeric_columns]
    
    # Vérifier s'il y a des valeurs NaN
    if X_train_numeric.isna().any().any() or y_train.isna().any():
        print("ATTENTION: Valeurs NaN détectées dans les données d'entraînement. Remplacement par 0.")
        X_train_numeric = X_train_numeric.fillna(0)
        y_train = y_train.fillna(0)
    
    if X_test_numeric.isna().any().any() or y_test.isna().any():
        print("ATTENTION: Valeurs NaN détectées dans les données de test. Remplacement par 0.")
        X_test_numeric = X_test_numeric.fillna(0)
        y_test = y_test.fillna(0)
    
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train_numeric, y_train)
    
    # Prédictions
    y_train_pred = model.predict(X_train_numeric)
    y_test_pred = model.predict(X_test_numeric)
    
    # Calcul des métriques
    metrics = {
        'train_mse': mean_squared_error(y_train, y_train_pred),
        'test_mse': mean_squared_error(y_test, y_test_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred)
    }
    
    # Affichage des métriques
    print(f"MSE (entraînement): {metrics['train_mse']:.2f}")
    print(f"MSE (test): {metrics['test_mse']:.2f}")
    print(f"RMSE (entraînement): {metrics['train_rmse']:.2f}")
    print(f"RMSE (test): {metrics['test_rmse']:.2f}")
    print(f"R² (entraînement): {metrics['train_r2']:.2f}")
    print(f"R² (test): {metrics['test_r2']:.2f}")
    
    # Feature importance
    print("\nImportance des features:")
    feature_importance = pd.DataFrame({
        'Feature': numeric_columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print(feature_importance.head(10))
    
    # Sauvegarde du modèle
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/random_forest.pkl")
    print("Modèle sauvegardé dans 'models/random_forest.pkl'")
    
    return model, metrics, y_test_pred

def visualize_predictions(y_true, y_pred, title):
    """Visualise les prédictions vs les valeurs réelles."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Ligne y=x (prédictions parfaites)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Prix réels')
    plt.ylabel('Prix prédits')
    plt.title(title)
    plt.grid(True)
    
    # Ajouter les métriques
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    plt.text(0.05, 0.95, f'RMSE = {rmse:.2f}\nR² = {r2:.2f}', 
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Créer le dossier results s'il n'existe pas
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{title.replace(' ', '_').lower()}.png")
    
    return plt

def compare_predictions(y_pred_simple, y_pred_multiple, title="Comparaison des prédictions"):
    """Compare les prédictions de deux modèles."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred_simple, y_pred_multiple, alpha=0.5)
    
    # Ligne y=x
    min_val = min(y_pred_simple.min(), y_pred_multiple.min())
    max_val = max(y_pred_simple.max(), y_pred_multiple.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Prédictions du modèle simple')
    plt.ylabel('Prédictions du modèle multiple')
    plt.title(title)
    plt.grid(True)
    
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{title.replace(' ', '_').lower()}.png")
    
    return plt

def plot_correlation_matrix(X_train, title="Matrice de corrélation des features"):
    """Affiche la matrice de corrélation entre les features."""
    # Sélectionner uniquement les colonnes numériques
    numeric_columns = X_train.select_dtypes(include=['int64', 'float64']).columns
    print(f"\nCalcul de la matrice de corrélation pour {len(numeric_columns)} colonnes numériques")
    
    # Utiliser seulement les colonnes numériques
    X_train_numeric = X_train[numeric_columns]
    
    # Calculer la matrice de corrélation
    corr_matrix = X_train_numeric.corr()
    
    # Afficher la matrice avec une taille adaptée au nombre de colonnes
    plt.figure(figsize=(max(12, len(numeric_columns) * 0.5), max(10, len(numeric_columns) * 0.5)))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Pour afficher seulement la moitié inférieure
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', mask=mask)
    plt.title(title)
    plt.tight_layout()
    
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{title.replace(' ', '_').lower()}.png")
    
    return plt

def main():
    """Point d'entrée principal."""
    # Charger les données
    X_train, y_train, X_test, y_test = load_data(
        "data/splits/train.csv", 
        "data/splits/test.csv"
    )
    
    # Identifier une feature numérique pour la régression simple
    numeric_features = X_train.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_features) > 0:
        # Calculer les corrélations pour trouver la meilleure feature
        correlations = []
        for feature in numeric_features:
            corr = y_train.corr(X_train[feature])
            correlations.append((feature, corr))
        
        # Trier par corrélation absolue décroissante
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        simple_feature = correlations[0][0]
        print(f"Feature sélectionnée pour la régression simple: {simple_feature} (corrélation: {correlations[0][1]:.4f})")
        
        # Entraîner la régression linéaire simple
        model_simple, metrics_simple, y_pred_simple = train_simple_linear_regression(
            X_train, X_test, y_train, y_test, simple_feature
        )
        
        # Visualiser les résultats de la régression simple
        visualize_predictions(
            y_test, y_pred_simple, 
            f"Régression Linéaire Simple ({simple_feature})"
        )
    else:
        print("Aucune feature numérique trouvée pour la régression simple")
    
    # Entraîner la régression linéaire multiple
    model_multiple, metrics_multiple, y_pred_multiple = train_multiple_linear_regression(
        X_train, X_test, y_train, y_test
    )
    
    # Visualiser les résultats de la régression multiple
    visualize_predictions(
        y_test, y_pred_multiple, 
        "Régression Linéaire Multiple"
    )
    
    # Comparer les modèles
    print("\nComparaison des modèles:")
    if 'metrics_simple' in locals():
        comparison = pd.DataFrame({
            'Métrique': ['MSE (test)', 'RMSE (test)', 'R² (test)'],
            'Régression Simple': [
                metrics_simple['test_mse'],
                metrics_simple['test_rmse'],
                metrics_simple['test_r2']
            ],
            'Régression Multiple': [
                metrics_multiple['test_mse'],
                metrics_multiple['test_rmse'],
                metrics_multiple['test_r2']
            ]
        })
        print(comparison)
        
        # Comparer directement les prédictions
        compare_predictions(y_pred_simple, y_pred_multiple)
    
    # Analyser les corrélations entre features
    plot_correlation_matrix(X_train)
    
    # Entraîner un modèle Random Forest
    model_rf, metrics_rf, y_pred_rf = train_random_forest(
        X_train, X_test, y_train, y_test
    )
    
    # Visualiser les résultats du Random Forest
    visualize_predictions(
        y_test, y_pred_rf, 
        "Random Forest"
    )
    
    # Comparer les trois modèles
    if 'metrics_simple' in locals():
        comparison_updated = pd.DataFrame({
            'Métrique': ['RMSE (test)', 'R² (test)'],
            f'Régression Simple ({simple_feature})': [
                metrics_simple['test_rmse'], 
                metrics_simple['test_r2']
            ],
            'Régression Multiple': [
                metrics_multiple['test_rmse'], 
                metrics_multiple['test_r2']
            ],
            'Random Forest': [
                metrics_rf['test_rmse'], 
                metrics_rf['test_r2']
            ]
        })
        print("\nComparaison des trois modèles:")
        print(comparison_updated)
    
if __name__ == "__main__":
    main()