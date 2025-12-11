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
OUTPUT_IMAGE = "stress_test_report.png"

def generate_high_risk_profiles(n=100, columns=None):
    """Génère n profils À HAUT RISQUE (Stress Test)."""
    print(f"Génération de {n} profils 'Pessimistes' (Burnout, bas salaire, insatisfaction)...")
    
    data = {
        # Jeunes, mobiles
        'Age': np.random.randint(19, 30, n),
        
        # Salaire bas (skew vers 25k-40k)
        'MonthlyIncome': np.random.normal(30000, 5000, n),
        
        # Expérience débutante
        'TotalWorkingYears': np.random.randint(1, 8, n),
        'YearsAtCompany': np.random.randint(0, 3, n),
        'YearsWithCurrManager': np.random.randint(0, 1, n), # Nouveau manager ou rotation fréquente
        
        # Conditions difficiles
        'DistanceFromHome': np.random.randint(20, 60, n), # Loin
        'MeanWorkingHours': np.random.normal(10.5, 1.0, n), # Grosses journées (>10h)
        'WorkDays': np.random.choice([250, 260, 280], n), # Travaille bcp
        'OverTimeFrequency': np.random.beta(5, 1, n), # Skew vers 1.0 (Toujours en heure sup)
        
        # Insatisfaction totale
        'EnvironmentSatisfaction': np.random.randint(1, 2, n),
        'JobSatisfaction': np.random.randint(1, 2, n),
        'WorkLifeBalance': np.random.randint(1, 2, n),
        
        # Facteurs aggravants
        'BusinessTravel_Travel_Frequently': np.ones(n), # Voyage tout le temps
        'MaritalStatus_Single': np.ones(n), # Célibataire (statistiquement plus mobile)
        'OverTime_Yes': np.ones(n) # Oui officiel
    }
    
    # Règles logiques
    data['YearsAtCompany'] = np.minimum(data['YearsAtCompany'], data['TotalWorkingYears'])
    data['YearsWithCurrManager'] = np.minimum(data['YearsWithCurrManager'], data['YearsAtCompany'])
    data['MonthlyIncome'] = np.maximum(data['MonthlyIncome'], 10000)
    
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
        
    for f in files:
        name = f.replace("attrition_model_", "").replace(".pkl", "")
        if name == "attrition_model": name = "RandomForest (Default)"
        
        path = os.path.join(MODEL_DIR, f)
        try:
            models[name] = joblib.load(path)
        except Exception:
            pass
            
    return models

def run_stress_test():
    print("--- STRESS TEST: SIMULATION PESSIMISTE ---")
    
    try:
        n_input = input("Combien d'employés 'à risque' simuler ? (défaut: 1000) : ")
        n_samples = int(n_input) if n_input.strip() else 1000
    except ValueError:
        n_samples = 1000
        
    models = load_all_models()
    try:
        model_columns = joblib.load(COLUMNS_PATH)
    except FileNotFoundError:
        print("Erreur: columns.pkl introuvable.")
        return

    # GÉNÉRATION PESSIMISTE
    X_sim = generate_high_risk_profiles(n_samples, model_columns)
    
    results = pd.DataFrame()
    print("\nCalcul des prédictions sur profils High-Risk...")
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_sim)[:, 1]
        else:
            probs = model.predict(X_sim)
        results[name] = probs

    # VISUALISATION (Même format que Comparative Simulation)
    print(f"\nGénération du rapport stress test ({OUTPUT_IMAGE})...")
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(22, 12))
    plt.suptitle(f"Rapport STRESS TEST : Performance sur Profils 'Burnout' (n={n_samples})", fontsize=20, weight='bold', color='darkred')

    # --- SECTION 1 : VUE "RH / BUSINESS" (Risques Employés) ---
    
    # A. Stratification des Risques (Low/Medium/High)
    plt.subplot(2, 2, 1)
    
    buckets = pd.DataFrame()
    for col in results.columns:
        low = (results[col] < 0.3).mean() * 100
        medium = ((results[col] >= 0.3) & (results[col] < 0.7)).mean() * 100
        high = (results[col] >= 0.7).mean() * 100
        buckets[col] = [low, medium, high]
    
    buckets.index = ['Faible (<30%)', 'Moyen (30-70%)', 'Élevé (>70%)']
    bucket_colors = ['#2ecc71', '#f1c40f', '#e74c3c'] 
    
    ax1 = plt.gca()
    buckets.T.plot(kind='bar', stacked=True, ax=ax1, color=bucket_colors, width=0.7)
    
    plt.title("Répartition des Populations par Niveau de Risque (Attendu : 100% Élevé)", fontsize=14)
    plt.ylabel("Pourcentage des employés (%)")
    plt.xlabel("")
    plt.xticks(rotation=45)
    plt.legend(title="Niveau de Risque", loc='upper left', bbox_to_anchor=(1, 1))
    
    # B. Matrice de Consensus
    plt.subplot(2, 2, 2)
    corr = results.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, cmap="Reds", vmin=0, vmax=1, square=True, fmt=".2f", linewidths=.5)
    plt.title("Consensus des Modèles sur le Danger", fontsize=14)

    # --- SECTION 2 : VUE "DATA SCIENCE" (Performance Technique) ---
    
    # C. Distribution Fine (KDE) + Métriques
    plt.subplot(2, 2, 3)
    
    metrics_text = {
        'rf': ' (Acc:99%, Recall:94%)',
        'xgboost': ' (Acc:99%, Recall:96%)',
        'logreg': ' (Acc:77%, Recall:77%)',
        'nb': ' (Acc:84%, Recall:20%)'
    }
    
    for col in results.columns:
        label_suffix = ""
        for key, val in metrics_text.items():
            if key in col.lower() or (key == 'rf' and 'random' in col.lower()):
                label_suffix = val
                break
                
        sns.kdeplot(results[col], label=f"{col}{label_suffix}", fill=True, alpha=0.1, linewidth=2)
        
    plt.title("Sensibilité des Modèles (Attendu : Pic à droite > 0.8)", fontsize=14)
    plt.xlabel("Probabilité de Départ Calculée")
    plt.ylabel("Fréquence d'apparition")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.xlim(0, 1)

    # D. Volume d'Alertes (>50%)
    plt.subplot(2, 2, 4)
    alerts = (results > 0.5).sum()
    alert_df = pd.DataFrame({'Modèle': alerts.index, 'Alertes': alerts.values})
    
    # Fix Warning: assign x to hue implicitly via palette usage or proper mapping
    barplot = sns.barplot(data=alert_df, x='Modèle', y='Alertes', hue='Modèle', palette="magma", legend=False)
    
    for i, p in enumerate(barplot.patches):
        barplot.annotate(f"{int(p.get_height())}\n({p.get_height()/n_samples:.1%})", 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha = 'center', va = 'bottom', xytext = (0, 5), 
                         textcoords = 'offset points', weight='bold')
                         
    plt.title(f"Nombre de détections réussies (Seuil > 50%) sur {n_samples} cas critiques", fontsize=14)
    plt.ylabel("Nombre d'employés flaggués")
    plt.xlabel("")
    plt.xticks(rotation=45)
    plt.ylim(0, n_samples * 1.2) 

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"Rapport sauvegardé : {OUTPUT_IMAGE}")
    
    # Résumé texte
    print("\n--- RÉSUMÉ DES DÉTECTIONS ---")
    print(alerts)
    print("-" * 30)

if __name__ == "__main__":
    run_stress_test()
