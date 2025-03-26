import pandas as pd
import numpy as np

def encode_categorical(df, columns):
    """Encode les colonnes catégorielles avec one-hot encoding."""
    return pd.get_dummies(df, columns=columns)

def create_price_ranges(df, column='price', bins=5):
    """Crée des intervalles de prix pour faciliter l'analyse."""
    df['price_range'] = pd.cut(df[column], bins=bins, labels=[f'range_{i+1}' for i in range(bins)])
    return df

def extract_numeric_features(df):
    """Extrait des features numériques supplémentaires si nécessaires."""
    # Exemple: créer un ratio si vous avez d'autres colonnes numériques
    if 'minimum_nights' in df.columns and df['minimum_nights'].min() > 0:
        df['price_per_night'] = df['price'] / df['minimum_nights']
    return df