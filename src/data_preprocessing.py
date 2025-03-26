import pandas as pd
import os

def select_columns(df, columns):
    """Sélectionne les colonnes spécifiées."""
    return df[columns]

def remove_duplicates(df, id_column='id'):
    """Supprime les doublons basés sur une colonne ID."""
    return df.drop_duplicates(subset=[id_column])

def handle_missing_values(df):
    """Gère les valeurs manquantes."""
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    return df

def convert_column_types(df):
    """Convertit les types de colonnes."""
    # Supprimer le symbole $ et la virgule de la colonne price
    if 'price' in df.columns and isinstance(df['price'].iloc[0], str):
        df['price'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float)
    elif 'price' in df.columns:
        df['price'] = df['price'].astype(float)
    
    # Convertir les types catégoriels
    if 'room_type' in df.columns:
        df['room_type'] = df['room_type'].astype('category')
    if 'property_type' in df.columns:
        df['property_type'] = df['property_type'].astype('category')
    if 'host_is_superhost' in df.columns:
        df['host_is_superhost'] = df['host_is_superhost'].map({'t': 1, 'f': 0, True: 1, False: 0}).fillna(0).astype(int)
    
    # Convertir les colonnes numériques
    numeric_cols = ['minimum_nights', 'number_of_reviews', 'accommodates', 'bedrooms', 'beds', 'review_scores_rating']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def preprocess_data(file_path, output_path):
    """Exécute toutes les étapes de prétraitement."""
    # S'assurer que le répertoire de sortie existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Lecture du fichier: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Nombre de lignes chargées: {len(df)}")
    
    # Liste étendue des colonnes pertinentes
    relevant_columns = [
        'id', 'price', 'neighbourhood_cleansed', 'room_type', 'minimum_nights',
        'name', 'latitude', 'longitude', 'number_of_reviews', 'accommodates', 
        'bedrooms', 'beds', 'review_scores_rating', 'host_is_superhost', 'property_type'
    ]
    
    # Vérifier que toutes les colonnes existent
    for col in relevant_columns[:]:
        if col not in df.columns:
            print(f"Attention: colonne '{col}' non trouvée dans le jeu de données")
            relevant_columns.remove(col)
    
    print(f"Colonnes sélectionnées: {relevant_columns}")
    df = df[relevant_columns]
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = convert_column_types(df)
    df.to_csv(output_path, index=False)
    print(f"Données prétraitées sauvegardées dans {output_path}")
    print(f"Nombre de lignes après traitement: {len(df)}")
    
    return df

def remove_outliers(df, column):
    """Supprime les valeurs aberrantes avec la méthode IQR."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    print(f"Détection d'outliers dans '{column}':")
    print(f"Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")
    print(f"Limites: [{lower_bound}, {upper_bound}]")
    print(f"Nombre d'outliers: {len(df[(df[column] < lower_bound) | (df[column] > upper_bound)])}")
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

if __name__ == "__main__":
    # Utiliser le fichier listings.csv au lieu de paris_listings.csv
    preprocess_data("data/raw/listings.csv", "data/processed/paris_listings_cleaned.csv")
