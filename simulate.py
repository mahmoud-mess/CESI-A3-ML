import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# --- CONFIGURATION ---
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "attrition_model.pkl")
COLUMNS_PATH = os.path.join(MODEL_DIR, "columns.pkl")
OUTPUT_IMAGE = "simulation_report.png"
N_SAMPLES = 100

def list_and_select_model():
    if not os.path.exists(MODEL_DIR):
        print(f"Dossier {MODEL_DIR} introuvable.")
        sys.exit(1)
        
    files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl') and f not in ['columns.pkl', 'encoders.pkl']]
    if not files:
        print(f"Aucun modèle trouvé dans {MODEL_DIR}. Veuillez lancer train_model.py.")
        sys.exit(1)
        
    print("\n--- CHOIX DU MODÈLE POUR SIMULATION ---")
    files.sort()
    for i, f in enumerate(files):
        print(f"{i+1}. {f}")
        
    while True:
        choice = input(f"Votre choix (1-{len(files)}) : ")
        if choice.isdigit() and 1 <= int(choice) <= len(files):
            return os.path.join(MODEL_DIR, files[int(choice)-1])
        print("Choix invalide.")

def generate_random_profiles(n=100, columns=None):
    """Génère n profils réalistes."""
    print(f"Génération de {n} profils aléatoires...")
    
    data = {
        'Age': np.random.randint(20, 60, n),
        'MonthlyIncome': np.random.lognormal(mean=10.8, sigma=0.6, size=n), # Loi log-normale pour salaires
        'TotalWorkingYears': np.random.randint(0, 40, n),
        'YearsAtCompany': np.random.randint(0, 20, n),
        'YearsWithCurrManager': np.random.randint(0, 15, n),
        'DistanceFromHome': np.random.randint(1, 30, n),
        'MeanWorkingHours': np.random.normal(8, 1.5, n), # Moyenne 8h, ecart-type 1.5h
        'WorkDays': np.random.randint(200, 260, n),
        'OverTimeFrequency': np.random.beta(2, 5, n), # Distribution Beta pour avoir plus de valeurs proches de 0
        'EnvironmentSatisfaction': np.random.randint(1, 5, n),
        'JobSatisfaction': np.random.randint(1, 5, n),
        'WorkLifeBalance': np.random.randint(1, 5, n),
        # Catégorielles simplifiées (on prend les plus courantes pour la simulation)
        'BusinessTravel_Travel_Frequently': np.random.choice([0, 1], n, p=[0.8, 0.2]),
        'MaritalStatus_Single': np.random.choice([0, 1], n, p=[0.6, 0.4]),
        'OverTime_Yes': np.random.choice([0, 1], n, p=[0.7, 0.3])
    }
    
    # Ajustements logiques
    # On ne peut pas avoir plus d'années en entreprise que d'années totales
    data['YearsAtCompany'] = np.minimum(data['YearsAtCompany'], data['TotalWorkingYears'])
    # On ne peut pas avoir plus d'années avec le manager que d'années en entreprise
    data['YearsWithCurrManager'] = np.minimum(data['YearsWithCurrManager'], data['YearsAtCompany'])
    # Salaire minimum
    data['MonthlyIncome'] = np.maximum(data['MonthlyIncome'], 20000)
    
    df = pd.DataFrame(data)
    
    # Remplir les autres colonnes manquantes avec des 0 pour respecter le format du modèle
    if columns:
        for col in columns:
            if col not in df.columns:
                df[col] = 0
                
        # Réordonner les colonnes (IMPORTANT pour que XGBoost/LogReg ne se plaignent pas)
        df = df[columns]
        
    return df

def simulate_and_visualize():
    # 1. Sélection du modèle
    print("--- SIMULATION D'ATTRITION ---")
    model_path = list_and_select_model()
    
    # Détermination du nom de sortie
    model_filename = os.path.basename(model_path)
    model_name_clean = os.path.splitext(model_filename)[0].replace("attrition_model_", "")
    if model_name_clean == "attrition_model": model_name_clean = "rf_default"
    
    output_image = f"simulation_report_{model_name_clean}.png"
    
    print(f"Chargement du modèle : {model_path}")
    
    if not os.path.exists(model_path):
        print("Modèle non trouvé.")
        return

    model = joblib.load(model_path)
    try:
        model_columns = joblib.load(COLUMNS_PATH)
    except FileNotFoundError:
        print("Erreur: columns.pkl introuvable. Entraînez un modèle d'abord.")
        return
    
    # 2. Génération
    X_sim = generate_random_profiles(N_SAMPLES, model_columns)
    
    # 3. Prédiction
    print("Calcul des risques...")
    # Gestion proba Pipeline vs Model
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_sim)[:, 1]
    else:
        probabilities = np.zeros(len(X_sim)) # Fallback
        
    predictions = model.predict(X_sim)
    
    X_sim['Risk_Probability'] = probabilities
    X_sim['Prediction'] = predictions
    
    # 4. Visualisation
    print(f"Génération des graphiques dans {output_image}...")
    plt.figure(figsize=(18, 10))
    
    # A. Histogramme des Risques
    plt.subplot(2, 2, 1)
    sns.histplot(probabilities, bins=20, kde=True, color='skyblue')
    plt.axvline(0.5, color='red', linestyle='--', label='Seuil de Départ (50%)')
    plt.title(f'Distribution des Probabilités ({model_name_clean})')
    plt.xlabel('Probabilité de Départ')
    plt.ylabel('Nombre d\'Employés')
    plt.legend()
    
    # B. Scatter Plot : Heures vs Ancienneté (Coloré par Risque)
    plt.subplot(2, 2, 2)
    # Check si colonnes existent (si features selectionnées)
    if 'YearsAtCompany' in X_sim.columns and 'MeanWorkingHours' in X_sim.columns:
        scatter = plt.scatter(
            X_sim['YearsAtCompany'], 
            X_sim['MeanWorkingHours'], 
            c=probabilities, 
            cmap='RdYlGn', # Rouge -> Vert (Att: Risk high = bad = red)
            # wait, RdYlGn : Red(low) -> Green(high).
            # Nous on veut Vert(low risk) -> Rouge(high risk).
            # Donc RdYlGn_r (Red Yellow Green Reversed) ?
            # 0 (Low risk) -> Green. 1 (High risk) -> Red.
            # RdYlGn : 0=Rouge, 1=Vert.
            # Donc on veut RdYlGn inversé: RdYlGn_r.
            # (Note: dans le code précédent c'était RdYlGn_r et je vois "Vert -> Rouge" en comm)
            # On garde RdYlGn_r
            cmap='RdYlGn_r', 
            s=100, 
            edgecolors='grey',
            alpha=0.8
        )
        plt.colorbar(scatter, label='Risque de Départ')
        plt.title('Carte des Risques : Ancienneté vs Heures')
        plt.xlabel('Années dans l\'Entreprise')
        plt.ylabel('Heures de Travail Moyennes / Jour')
        plt.grid(True, alpha=0.3)
    
    # C. Boxplot : Satisfaction vs Risque
    plt.subplot(2, 2, 3)
    if 'JobSatisfaction' in X_sim.columns:
        sns.boxplot(x='JobSatisfaction', y='Risk_Probability', data=X_sim, palette="Set2")
        plt.title('Impact de la Satisfaction au Travail sur le Risque')
        plt.xlabel('Niveau de Satisfaction (1-4)')
        plt.ylabel('Probabilité de Départ')
    
    # D. Scatter Plot : Salaire vs Age
    plt.subplot(2, 2, 4)
    if 'Age' in X_sim.columns and 'MonthlyIncome' in X_sim.columns:
        scatter2 = plt.scatter(
            X_sim['Age'], 
            X_sim['MonthlyIncome'], 
            c=probabilities, 
            cmap='RdYlGn_r', 
            s=100, 
            edgecolors='grey',
            alpha=0.8
        )
        plt.colorbar(scatter2, label='Risque de Départ')
        plt.title('Carte des Risques : Age vs Salaire')
        plt.xlabel('Age')
        plt.ylabel('Salaire Mensuel')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"Rapport de simulation sauvegardé : {output_image}")

if __name__ == "__main__":
    simulate_and_visualize()
