"""
Module de prétraitement des données pour le projet de prédiction des prix Airbnb.

Ce module contient toutes les fonctions nécessaires pour nettoyer, transformer
et préparer les données brutes d'Airbnb avant l'analyse et la modélisation.

Fonctions principales:
    * preprocess_data - Fonction principale qui exécute toutes les étapes de prétraitement
    * remove_outliers - Détecte et supprime les valeurs aberrantes
    * convert_column_types - Convertit les types de données des colonnes
    * handle_missing_values - Gère les valeurs manquantes dans le dataset
"""
import pandas as pd
import os


def select_columns(df, columns):
    """
    Sélectionne un sous-ensemble de colonnes du DataFrame.
    
    Args:
        df (pandas.DataFrame): Le DataFrame source
        columns (list): Liste des noms de colonnes à conserver
        
    Returns:
        pandas.DataFrame: DataFrame avec uniquement les colonnes sélectionnées
    """
    return df[columns]


def remove_duplicates(df, id_column='id'):
    """
    Supprime les entrées dupliquées basées sur une colonne d'identifiant.
    
    Args:
        df (pandas.DataFrame): Le DataFrame à nettoyer
        id_column (str, optional): Nom de la colonne d'identifiant unique. 
                                  Par défaut: 'id'
                                  
    Returns:
        pandas.DataFrame: DataFrame sans doublons
    """
    return df.drop_duplicates(subset=[id_column])


def handle_missing_values(df):
    """
    Gère les valeurs manquantes dans le DataFrame.
    
    Pour les colonnes numériques: remplit avec la médiane
    Pour les colonnes objet/string: remplit avec le mode (valeur la plus fréquente)
    
    Args:
        df (pandas.DataFrame): Le DataFrame avec des valeurs manquantes
        
    Returns:
        pandas.DataFrame: DataFrame avec les valeurs manquantes traitées
    """
    # Traitement des colonnes numériques
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col].fillna(df[col].median(), inplace=True)
    
    # Traitement des colonnes catégorielles
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df


def convert_column_types(df):
    """
    Convertit les types de données des colonnes pour assurer la cohérence du dataset.
    
    Opérations principales:
    - Conversion de la colonne prix (suppression des symboles monétaires)
    - Conversion des variables catégorielles
    - Conversion des booléens en entiers (0/1)
    - Conversion des colonnes numériques
    
    Args:
        df (pandas.DataFrame): DataFrame à transformer
        
    Returns:
        pandas.DataFrame: DataFrame avec les types de données corrects
    """
    # Conversion de la colonne prix
    if 'price' in df.columns and isinstance(df['price'].iloc[0], str):
        df['price'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float)
    elif 'price' in df.columns:
        df['price'] = df['price'].astype(float)
    
    # Conversion des types catégoriels pour optimiser la mémoire
    if 'room_type' in df.columns:
        df['room_type'] = df['room_type'].astype('category')
    if 'property_type' in df.columns:
        df['property_type'] = df['property_type'].astype('category')
    
    # Conversion des indicateurs booléens en entiers binaires (0/1)
    if 'host_is_superhost' in df.columns:
        df['host_is_superhost'] = df['host_is_superhost'].map({'t': 1, 'f': 0, True: 1, False: 0}).fillna(0).astype(int)
    
    # Conversion des colonnes numériques avec gestion d'erreurs
    numeric_cols = ['minimum_nights', 'number_of_reviews', 'accommodates', 'bedrooms', 'beds', 'review_scores_rating']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # 'coerce' convertit les erreurs en NaN
    
    return df


def preprocess_data(file_path, output_path):
    """
    Fonction principale qui exécute l'ensemble du pipeline de prétraitement.
    
    Étapes:
    1. Chargement des données brutes
    2. Sélection des colonnes pertinentes
    3. Suppression des doublons
    4. Traitement des valeurs manquantes
    5. Conversion des types de données
    6. Sauvegarde des données nettoyées
    
    Args:
        file_path (str): Chemin vers le fichier CSV brut
        output_path (str): Chemin où sauvegarder le fichier nettoyé
        
    Returns:
        pandas.DataFrame: DataFrame prétraité
        
    Example:
        >>> preprocess_data("data/raw/listings.csv", "data/processed/cleaned_data.csv")
    """
    # Création du répertoire de sortie s'il n'existe pas
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Étape 1: Chargement des données
    print(f"Lecture du fichier: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Nombre de lignes chargées: {len(df)}")
    
    # Étape 2: Définition et sélection des colonnes pertinentes
    # Liste étendue des colonnes pertinentes pour l'analyse de prix
    relevant_columns = [
        'id', 'price', 'neighbourhood_cleansed', 'room_type', 'minimum_nights',
        'name', 'latitude', 'longitude', 'number_of_reviews', 'accommodates', 
        'bedrooms', 'beds', 'review_scores_rating', 'host_is_superhost', 'property_type'
    ]
    
    # Vérification de l'existence des colonnes dans le dataset
    for col in relevant_columns[:]:  # Utilisation d'une copie pour itération sécurisée
        if col not in df.columns:
            print(f"Attention: colonne '{col}' non trouvée dans le jeu de données")
            relevant_columns.remove(col)
    
    print(f"Colonnes sélectionnées: {relevant_columns}")
    df = df[relevant_columns]
    
    # Étapes 3-5: Application des fonctions de nettoyage
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = convert_column_types(df)
    
    # Étape 6: Sauvegarde des résultats
    df.to_csv(output_path, index=False)
    print(f"Données prétraitées sauvegardées dans {output_path}")
    print(f"Nombre de lignes après traitement: {len(df)}")
    
    return df


def remove_outliers(df, column):
    """
    Supprime les valeurs aberrantes en utilisant la méthode de l'écart interquartile (IQR).
    
    Formule:
    - Limite inférieure = Q1 - 1.5 * IQR
    - Limite supérieure = Q3 + 1.5 * IQR
    où IQR = Q3 - Q1
    
    Args:
        df (pandas.DataFrame): DataFrame contenant les données
        column (str): Nom de la colonne à nettoyer des outliers
        
    Returns:
        pandas.DataFrame: DataFrame sans les valeurs aberrantes pour la colonne spécifiée
        
    Example:
        >>> clean_df = remove_outliers(df, 'price')
    """
    # Calcul des quartiles et de l'IQR
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    # Définition des limites pour la détection d'outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Affichage des statistiques de détection
    print(f"Détection d'outliers dans '{column}':")
    print(f"Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")
    print(f"Limites: [{lower_bound}, {upper_bound}]")
    print(f"Nombre d'outliers: {len(df[(df[column] < lower_bound) | (df[column] > upper_bound)])}")
    
    # Filtrage des données pour exclure les outliers
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


if __name__ == "__main__":
    # Point d'entrée du script quand exécuté directement
    preprocess_data("data/raw/listings.csv", "data/processed/paris_listings_cleaned.csv")
