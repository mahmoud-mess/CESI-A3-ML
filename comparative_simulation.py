import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# --- CONFIGURATION ---
MODEL_DIR = "models"
COLUMNS_PATH = os.path.join(MODEL_DIR, "columns.pkl")
OUTPUT_IMAGE = "comparative_simulation_report.png"

def generate_random_profiles(n=100, columns=None):
    """Génère n profils réalistes (Copie de simulate.py pour indépendance)."""
    print(f"Génération de {n} profils aléatoires...")
    
    data = {
        'Age': np.random.randint(20, 60, n),
        'MonthlyIncome': np.random.lognormal(mean=10.8, sigma=0.6, size=n),
        'TotalWorkingYears': np.random.randint(0, 40, n),
        'YearsAtCompany': np.random.randint(0, 20, n),
        'YearsWithCurrManager': np.random.randint(0, 15, n),
        'DistanceFromHome': np.random.randint(1, 30, n),
        'MeanWorkingHours': np.random.normal(8, 1.5, n),
        'WorkDays': np.random.randint(200, 260, n),
        'OverTimeFrequency': np.random.beta(2, 5, n),
        'EnvironmentSatisfaction': np.random.randint(1, 5, n),
        'JobSatisfaction': np.random.randint(1, 5, n),
        'WorkLifeBalance': np.random.randint(1, 5, n),
        # Catégorielles
        'BusinessTravel_Travel_Frequently': np.random.choice([0, 1], n, p=[0.8, 0.2]),
        'MaritalStatus_Single': np.random.choice([0, 1], n, p=[0.6, 0.4]),
        'OverTime_Yes': np.random.choice([0, 1], n, p=[0.7, 0.3])
    }
    
    # Règles logiques
    data['YearsAtCompany'] = np.minimum(data['YearsAtCompany'], data['TotalWorkingYears'])
    data['YearsWithCurrManager'] = np.minimum(data['YearsWithCurrManager'], data['YearsAtCompany'])
    data['MonthlyIncome'] = np.maximum(data['MonthlyIncome'], 20000)
    
    df = pd.DataFrame(data)
    
    if columns:
        for col in columns:
            if col not in df.columns:
                df[col] = 0
        df = df[columns]
        
    return df

def load_all_models():
    """Charge tous les modèles .pkl disponibles."""
    if not os.path.exists(MODEL_DIR):
        print(f"Dossier {MODEL_DIR} introuvable.")
        sys.exit(1)
        
    models = {}
    files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl') and f not in ['columns.pkl', 'encoders.pkl']]
    
    if not files:
        print("Aucun modèle trouvé.")
        sys.exit(1)
        
    print(f"Chargement de {len(files)} modèles...")
    for f in files:
        # Simplification du nom (attrition_model_xgboost.pkl -> xgboost)
        name = f.replace("attrition_model_", "").replace(".pkl", "")
        if name == "attrition_model": name = "RandomForest (Default)"
        
        path = os.path.join(MODEL_DIR, f)
        try:
            models[name] = joblib.load(path)
            print(f" - {name} chargé.")
        except Exception as e:
            print(f" ! Erreur chargement {name}: {e}")
            
    return models

def run_simulation():
    print("--- SIMULATION COMPARATIVE INSTRUCTIVE ---")
    
    # 1. Configuration
    try:
        n_input = input("Combien d'employés simuler ? (défaut: 500) : ")
        n_samples = int(n_input) if n_input.strip() else 500
    except ValueError:
        n_samples = 500
        print("Erreur de saisie, utilisation de 500 échantillons.") # Fixed typo 'd' -> 'de'
        
    # 2. Chargement des ressources
    models = load_all_models()
    try:
        model_columns = joblib.load(COLUMNS_PATH)
    except FileNotFoundError:
        print("Erreur: columns.pkl introuvable.")
        return

    # 3. Génération des données
    X_sim = generate_random_profiles(n_samples, model_columns)
    
    # 4. Predictions pour chaque modèle
    results = pd.DataFrame()
    
    print("\nCalcul des prédictions...")
    for name, model in models.items():
        # Gestion proba
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_sim)[:, 1]
        else:
            probs = model.predict(X_sim) # Fallback binaire
            
        results[name] = probs

    # 5. Visualisation
    print(f"\nGénération du rapport comparatif ({OUTPUT_IMAGE})...")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(22, 12))
    plt.suptitle("Rapport de Simulation : Impact Employés vs Performance ML", fontsize=20, weight='bold')

    # --- SECTION 1 : VUE "RH / BUSINESS" (Risques Employés) ---
    
    # A. Stratification des Risques (Low/Medium/High)
    plt.subplot(2, 2, 1)
    
    # Création des buckets
    buckets = pd.DataFrame()
    for col in results.columns:
        low = (results[col] < 0.3).mean() * 100
        medium = ((results[col] >= 0.3) & (results[col] < 0.7)).mean() * 100
        high = (results[col] >= 0.7).mean() * 100
        buckets[col] = [low, medium, high]
    
    buckets.index = ['Faible (<30%)', 'Moyen (30-70%)', 'Élevé (>70%)']
    bucket_data = buckets.T.reset_index().melt(id_vars='index')
    bucket_data.columns = ['Modèle', 'Niveau de Risque', 'Pourcentage']
    
    # Visualisation via Barplot Empilé (plus propre que histplot)
    bucket_colors = ['#2ecc71', '#f1c40f', '#e74c3c'] # Vert, Jaune, Rouge
    
    # On plotte "Low + Medium + High" (Fond), puis "Medium + High", puis "High" pour simuler l'empilement
    # Ou mieux: Pandas plot bar stacked
    ax1 = plt.gca()
    buckets.T.plot(kind='bar', stacked=True, ax=ax1, color=bucket_colors, width=0.7)
    
    plt.title("Répartition des Populations par Niveau de Risque", fontsize=14)
    plt.ylabel("Pourcentage des employés (%)")
    plt.xlabel("")
    plt.xticks(rotation=45)
    plt.legend(title="Niveau de Risque", loc='upper left', bbox_to_anchor=(1, 1))
    
    # B. Matrice de Consensus
    plt.subplot(2, 2, 2)
    corr = results.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, cmap="Blues", vmin=0, vmax=1, square=True, fmt=".2f", linewidths=.5)
    plt.title("Consensus des Modèles (Corrélation)", fontsize=14)

    # --- SECTION 2 : VUE "DATA SCIENCE" (Performance Technique) ---
    
    # C. Distribution Fine (KDE) + Métriques
    plt.subplot(2, 2, 3)
    
    # Métriques connues (Hardcodées suite à l'analyse précédente pour contexte)
    metrics_text = {
        'rf': ' (Acc:99%, Recall:94%)',
        'xgboost': ' (Acc:99%, Recall:96%)',
        'logreg': ' (Acc:77%, Recall:77%)',
        'nb': ' (Acc:84%, Recall:20%)'
    }
    
    for col in results.columns:
        # Trouver la métrique correspondante
        label_suffix = ""
        for key, val in metrics_text.items():
            if key in col.lower() or (key == 'rf' and 'random' in col.lower()):
                label_suffix = val
                break
                
        sns.kdeplot(results[col], label=f"{col}{label_suffix}", fill=True, alpha=0.1, linewidth=2)
        
    plt.title("Sensibilité des Modèles (Distribution de Densité)", fontsize=14)
    plt.xlabel("Probabilité de Départ Calculée")
    plt.ylabel("Fréquence d'apparition")
    plt.legend()
    plt.grid(True, alpha=0.2)

    # D. Volume d'Alertes (>50%)
    plt.subplot(2, 2, 4)
    alerts = (results > 0.5).sum()
    alert_df = pd.DataFrame({'Modèle': alerts.index, 'Alertes': alerts.values})
    
    # Fix Warning: assign x to hue implicitly via palette usage or proper mapping
    barplot = sns.barplot(data=alert_df, x='Modèle', y='Alertes', hue='Modèle', palette="viridis", legend=False)
    
    # Ajout des labels sur les barres
    for i, p in enumerate(barplot.patches):
        barplot.annotate(f"{int(p.get_height())}\n({p.get_height()/n_samples:.1%})", 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha = 'center', va = 'bottom', xytext = (0, 5), 
                         textcoords = 'offset points')
                         
    plt.title(f"Volume d'Alertes Actives (Seuil > 50%) sur {n_samples} employés", fontsize=14)
    plt.ylabel("Nombre d'employés flaggués")
    plt.xlabel("")
    plt.xticks(rotation=45)
    plt.ylim(0, alerts.max() * 1.2) # Marge pour le texte

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Marge pour le titre principal
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"Rapport sauvegardé : {OUTPUT_IMAGE}")
    
    # Résumé texte
    print("\n--- RÉSUMÉ DES ALERTES ---")
    print(alerts)
    print("-" * 30)

if __name__ == "__main__":
    run_simulation()
