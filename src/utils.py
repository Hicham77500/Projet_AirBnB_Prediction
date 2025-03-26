import os
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(df, target_column, test_size=0.2, random_state=42):
    """Divise les données en ensembles d'entraînement et de test."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def save_splits(X_train, X_test, y_train, y_test, output_dir="data/splits"):
    """Sauvegarde les ensembles d'entraînement et de test."""
    # Créer le répertoire s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Recombiner les features et les cibles
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    # Sauvegarder en CSV
    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)
    
    print(f"Ensembles de données sauvegardés dans {output_dir}")
    print(f"Train: {train_df.shape}, Test: {test_df.shape}")