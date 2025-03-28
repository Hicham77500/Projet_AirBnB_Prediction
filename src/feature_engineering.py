"""
Module pour l'ingénierie des features dans le projet de prédiction des prix Airbnb.

Ce module contient des fonctions pour :
- Encoder les variables catégorielles.
- Créer des intervalles de prix pour l'analyse.
- Extraire des features numériques supplémentaires.

Fonctions principales :
    - encode_categorical : Encode les colonnes catégorielles avec one-hot encoding.
    - create_price_ranges : Crée des intervalles de prix pour faciliter l'analyse.
    - extract_numeric_features : Extrait des features numériques supplémentaires.
"""

import pandas as pd
import numpy as np


def encode_categorical(df, columns):
    """
    Encode les colonnes catégorielles spécifiées avec la méthode one-hot encoding.

    Args:
        df (pd.DataFrame): DataFrame contenant les données.
        columns (list): Liste des colonnes catégorielles à encoder.

    Returns:
        pd.DataFrame: DataFrame avec les colonnes encodées.
    """
    return pd.get_dummies(df, columns=columns)


def create_price_ranges(df, column='price', bins=5):
    """
    Crée des intervalles de prix pour faciliter l'analyse.

    Args:
        df (pd.DataFrame): DataFrame contenant les données.
        column (str): Nom de la colonne contenant les prix.
        bins (int): Nombre d'intervalles à créer.

    Returns:
        pd.DataFrame: DataFrame avec une nouvelle colonne 'price_range'.
    """
    df['price_range'] = pd.cut(df[column], bins=bins, labels=[f'range_{i+1}' for i in range(bins)])
    return df


def extract_numeric_features(df):
    """
    Extrait des features numériques supplémentaires si nécessaires.

    Exemple : Création d'un ratio entre le prix et le nombre de nuits minimum.

    Args:
        df (pd.DataFrame): DataFrame contenant les données.

    Returns:
        pd.DataFrame: DataFrame avec les nouvelles features ajoutées.
    """
    if 'minimum_nights' in df.columns and df['minimum_nights'].min() > 0:
        df['price_per_night'] = df['price'] / df['minimum_nights']
    return df