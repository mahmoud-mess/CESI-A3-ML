import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
DATA_DIR = "."
OUTPUT_FILE = "processed_data.csv"

def load_time_data(file_path, prefix):
    """Charge et traite les fichiers de temps (in/out)."""
    print(f"Traitement de {file_path}...")
    # Le fichier contient une première colonne vide qui est l'ID implicite (1-based)
    df = pd.read_csv(file_path)
    
    # Renommer la première colonne pour être sûr
    df.rename(columns={df.columns[0]: 'EmployeeID'}, inplace=True)
    
    # Conversion en datetime pour toutes les colonnes sauf EmployeeID
    # On utilise apply pour aller plus vite, errors='coerce' gère les NaNs
    date_cols = df.columns[1:]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        
    return df

def calculate_time_features(in_df, out_df):
    """Calcule la durée de travail moyenne et autres métriques."""
    print("Calcul des métriques temporelles (cela peut prendre un moment)...")
    
    # S'assurer que les index sont alignés
    in_df = in_df.set_index('EmployeeID')
    out_df = out_df.set_index('EmployeeID')
    
    # Calcul de la durée (Out - In)
    # Pandas gère automatiquement l'alignement des colonnes et index
    duration_df = out_df - in_df
    
    # Convertir en heures (float)
    # timedelta / timedelta(hours=1)
    duration_hours = duration_df.apply(lambda x: x.dt.total_seconds() / 3600.0)
    
    # 1. Moyenne des heures travaillées par jour (en ignorant les absences/NaN)
    mean_hours = duration_hours.mean(axis=1)
    
    # 2. Nombre de jours travaillés (non-NaN)
    work_days = duration_hours.count(axis=1)
    
    # 3. Fréquence des heures supplémentaires (> 8h) - Optionnel mais utile
    # On crée un masque booléen, on somme les True, divisé par work_days
    overtime_freq = (duration_hours > 8.0).sum(axis=1) / work_days
    
    # Création du DataFrame final
    features = pd.DataFrame({
        'MeanWorkingHours': mean_hours,
        'WorkDays': work_days,
        'OverTimeFrequency': overtime_freq
    })
    
    # Remplir les NaN éventuels (si un employé n'a jamais travaillé, ce qui serait bizarre)
    features.fillna(0, inplace=True)
    
    return features

def main():
    print("--- DÉBUT DU TRAITEMENT DES DONNÉES ---")
    
    # 1. Chargement des données principales
    print("Chargement des données CSV principales...")
    general_data = pd.read_csv(os.path.join(DATA_DIR, "general_data.csv"))
    manager_survey = pd.read_csv(os.path.join(DATA_DIR, "manager_survey_data.csv"))
    employee_survey = pd.read_csv(os.path.join(DATA_DIR, "employee_survey_data.csv"))
    
    # 2. Fusion des données principales
    print("Fusion des datasets...")
    merged_df = general_data.merge(manager_survey, on='EmployeeID', how='left')
    merged_df = merged_df.merge(employee_survey, on='EmployeeID', how='left')
    
    # 3. Traitement des données temporelles
    in_time_path = os.path.join(DATA_DIR, "in_time.csv")
    out_time_path = os.path.join(DATA_DIR, "out_time.csv")
    
    if os.path.exists(in_time_path) and os.path.exists(out_time_path):
        in_df = load_time_data(in_time_path, "in")
        out_df = load_time_data(out_time_path, "out")
        
        time_features = calculate_time_features(in_df, out_df)
        
        # Reset index pour récupérer EmployeeID en colonne pour le merge
        time_features.reset_index(inplace=True)
        
        # Fusion avec le dataset principal
        merged_df = merged_df.merge(time_features, on='EmployeeID', how='left')
    else:
        print("ATTENTION : Fichiers de temps introuvables. Les métriques temporelles seront ignorées.")

    # 4. Nettoyage basique
    print("Nettoyage des données...")
    
    # Suppression des colonnes inutiles (constantes ou ID)
    cols_to_drop = ['EmployeeCount', 'Over18', 'StandardHours'] 
    # On garde EmployeeID pour l'instant pour le tracking, on l'enlèvera avant le training
    merged_df.drop(columns=[c for c in cols_to_drop if c in merged_df.columns], inplace=True)
    
    # Gestion des NaN
    # Pour les enquêtes (survey), NaN souvent = pas d'avis => on peut mettre la médiane (souvent 3) ou 0
    # Pour le dataset, on va opter pour une stratégie simple : imputation médiane pour le numérique, mode pour cat
    
    print(f"Valeurs manquantes avant nettoyage : {merged_df.isnull().sum().sum()}")
    
    for col in merged_df.columns:
        if merged_df[col].dtype == 'object':
            # Catégoriel : Remplacer par le mode
            merged_df[col] = merged_df[col].fillna(merged_df[col].mode()[0])
        else:
            # Numérique : Remplacer par la médiane (plus robuste que moyenne)
            merged_df[col] = merged_df[col].fillna(merged_df[col].median())

    print(f"Valeurs manquantes après nettoyage : {merged_df.isnull().sum().sum()}")

    # 5. Sauvegarde
    output_path = os.path.join(DATA_DIR, OUTPUT_FILE)
    merged_df.to_csv(output_path, index=False)
    print(f"Dataset traité sauvegardé sous : {output_path}")
    print(f"Dimensions finales : {merged_df.shape}")
    print("--- TRAITEMENT TERMINÉ ---")

if __name__ == "__main__":
    main()
