import pandas as pd
import numpy as np
import joblib
import os
import shap
import sys

# --- CONFIGURATION ---
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "attrition_model.pkl")
COLUMNS_PATH = os.path.join(MODEL_DIR, "columns.pkl")
# On a besoin d'un dataset de référence pour SHAP (background distribution)
# Idéalement on prendrait X_train, ici on va charger processed_data pour en extraire un échantillon
DATA_FILE = "processed_data.csv"

def get_user_input(model_columns):
    """Demande à l'utilisateur de saisir les valeurs pour les features principales."""
    print("\n--- Saisie des informations de l'employé ---")
    print("(Appuyez sur Entrée pour utiliser la valeur par défaut/médiane)")
    
    input_data = {}
    
    # Dictionnaire des valeurs par défaut
    defaults = {
        'Age': 35,
        'MonthlyIncome': 65000,
        'TotalWorkingYears': 10,
        'YearsAtCompany': 5,
        'YearsWithCurrManager': 3,
        'DistanceFromHome': 10,
        'MeanWorkingHours': 7.5,
        'WorkDays': 250,
        'OverTimeFrequency': 0.1, # Ceci est une valeur numérique dérivée, ne pas confondre avec la catégorielle 'OverTime'
        'EnvironmentSatisfaction': 3,
        'JobSatisfaction': 3,
        'WorkLifeBalance': 3,
        # Valeurs par défaut pour les catégories
        'BusinessTravel': 'Travel_Rarely',
        'Department': 'Research & Development',
        'EducationField': 'Life Sciences',
        'Gender': 'Male',
        'JobRole': 'Sales Executive',
        'MaritalStatus': 'Married',
        'OverTime': 'No' # C'est la catégorielle, pas la fréquence
    }

    # Liste des features numériques clés à demander
    key_numerical_features = [
        'Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany', 
        'YearsWithCurrManager', 'DistanceFromHome', 'MeanWorkingHours', 
        'WorkDays', 'OverTimeFrequency', # OverTimeFrequency est numérique ici
        'EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance'
    ]

    # Liste des features catégorielles clés à demander avec leurs options
    key_categorical_features = {
        'BusinessTravel': ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'],
        'Department': ['Research & Development', 'Sales', 'Human Resources'],
        'EducationField': ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Other', 'Human Resources'],
        'Gender': ['Female', 'Male'],
        'JobRole': ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'],
        'MaritalStatus': ['Single', 'Married', 'Divorced'],
        'OverTime': ['No', 'Yes']
    }
    
    # Validation des plages réalistes pour les numériques
    validation_rules = {
        'Age': (18, 70),
        'MonthlyIncome': (1000, 500000),
        'TotalWorkingYears': (0, 50),
        'YearsAtCompany': (0, 50),
        'YearsWithCurrManager': (0, 50),
        'DistanceFromHome': (0, 100),
        'MeanWorkingHours': (0, 24),
        'WorkDays': (0, 300),
        'OverTimeFrequency': (0.0, 1.0),
        'EnvironmentSatisfaction': (1, 4),
        'JobSatisfaction': (1, 4),
        'WorkLifeBalance': (1, 4)
    }

    # Saisie des numériques
    for feature in key_numerical_features:
        default_val = defaults.get(feature, 0)
        min_val, max_val = validation_rules.get(feature, (0, float('inf')))
        
        while True:
            val = input(f"{feature} (défaut: {default_val}) [{min_val}-{max_val}]: ")
            if val == "":
                input_data[feature] = default_val
                break
            try:
                float_val = float(val)
                if min_val <= float_val <= max_val:
                    input_data[feature] = float_val
                    break
                else:
                    print(f"Erreur : La valeur doit être comprise entre {min_val} et {max_val}.")
            except ValueError:
                print("Veuillez entrer un nombre valide.")
    
    # Saisie des catégorielles
    for feature, options in key_categorical_features.items():
        default_val = defaults.get(feature, options[0])
        print(f"\n--- {feature} (défaut: {default_val}) ---")
        for i, option in enumerate(options):
            print(f"{i+1}. {option}")
        
        while True:
            choice = input(f"Votre choix (1-{len(options)}, Entrée pour défaut): ")
            if choice == "":
                input_data[feature] = default_val
                break
            try:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(options):
                    input_data[feature] = options[choice_idx]
                    break
                else:
                    print(f"Erreur : Veuillez choisir un nombre entre 1 et {len(options)}.")
            except ValueError:
                print("Veuillez entrer un nombre valide.")
                
    return input_data

def prepare_input_dataframe(user_input, model_columns):
    """Crée un DataFrame prêt pour le modèle avec toutes les colonnes à 0 sauf celles saisies."""
    # Créer un DF avec une seule ligne remplie de zéros
    df = pd.DataFrame(0, index=[0], columns=model_columns)
    
    # Remplir avec les données utilisateur numériques
    for col, val in user_input.items():
        if col in df.columns: # Pour les colonnes numériques directes
            df[col] = val
        # Pour les colonnes catégorielles, on cherche la colonne one-hot encodée
        else: 
            # Reconstituer le nom de la colonne one-hot encodée
            # Exemple: BusinessTravel_Travel_Frequently
            if col == 'Gender' and val == 'Male': # Si Male, on active Gender_Male
                if 'Gender_Male' in df.columns:
                    df['Gender_Male'] = 1
            elif col == 'OverTime' and val == 'Yes': # Si Yes, on active OverTime_Yes
                if 'OverTime_Yes' in df.columns:
                    df['OverTime_Yes'] = 1
            else:
                dummy_col_name = f"{col}_{val}"
                if dummy_col_name in df.columns:
                    df[dummy_col_name] = 1
    
    return df

def list_and_select_model():
    files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl') and f not in ['columns.pkl', 'encoders.pkl']]
    if not files:
        print(f"Aucun modèle trouvé dans {MODEL_DIR}. Veuillez lancer train_model.py.")
        sys.exit(1)
        
    print("\n--- CHOIX DU MODÈLE ---")
    files.sort()
    for i, f in enumerate(files):
        print(f"{i+1}. {f}")
        
    while True:
        choice = input(f"Votre choix (1-{len(files)}) : ")
        if choice.isdigit() and 1 <= int(choice) <= len(files):
            return os.path.join(MODEL_DIR, files[int(choice)-1])
        print("Choix invalide.")

def predict_and_explain():
    # 1. Sélection et Chargement
    model_path = list_and_select_model()
    print(f"Chargement du modèle : {model_path}...")
    
    try:
        model = joblib.load(model_path)
        model_columns = joblib.load(COLUMNS_PATH)
    except FileNotFoundError:
        print("Erreur : Fichiers manquants (modèle ou columns.pkl).")
        sys.exit(1)
        
    # 2. Saisie utilisateur
    user_data = get_user_input(model_columns)
    input_df = prepare_input_dataframe(user_data, model_columns)
    
    # 3. Prédiction
    print("\n--- Analyse en cours... ---")
    prediction = model.predict(input_df)[0]
    
    # Gestion proba (certains modèles comme SVM n'ont pas predict_proba par défaut, mais ici on a LogReg/RF/XGB/NB qui l'ont)
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_df)[0][1]
    else:
        probability = 0.5 # Fallback ou 1.0/0.0 selon la prédiction
    
    print(f"\nRésultat : {'⚠️ RISQUE DE DÉPART (Oui)' if prediction == 1 else '✅ FIDÉLISATION PROBABLE (Non)'}")
    print(f"Probabilité de départ estimée : {probability:.1%}")
    
    # 4. Explication SHAP
    print("\nCalcul des facteurs d'influence (SHAP values)...")
    
    # Gestion Pipeline
    estimator = model
    explainer_input = input_df
    
    if hasattr(model, 'named_steps'):
        # C'est un pipeline
        estimator = model.steps[-1][1]
        preprocessor = model.steps[:-1] # Tout sauf le dernier
        
        # On transforme l'entrée pour SHAP (car l'explainer doit voir les données scalées)
        # Attention: Pipeline slice returns a Pipeline, we can fit_transform or just transform if already fitted
        # Le modèle global est déjà fitté, donc les steps aussi.
        # On applique les transformations séquentiellement
        for name, step in preprocessor:
             if hasattr(step, 'transform'):
                explainer_input = step.transform(explainer_input)
    
    # Détection du type de modèle pour l'explainer
    class_name = estimator.__class__.__name__
    explainer = None
    shap_values_obj = None  # Initialisation pour éviter UnboundLocalError
    
    # Background dataset nécessaire pour Linear et Kernel
    # On charge un petit échantillon et on le transforme de la même façon si pipeline
    try:
        if os.path.exists(DATA_FILE):
             # Chargement partiel pour la baseline
             df_bg = pd.read_csv(DATA_FILE).sample(100, random_state=42)
             # Ici c'est compliqué car on a besoin que df_bg ait les mêmes colonnes que input_df (one-hot encoded)
             # Or DATA_FILE est brut ou processed? DATA_FILE est "processed_data.csv" qui est déjà clean mais pas one-hot pour l'affichage?
             # Ah, DATA_FILE a déjà les features numériques/catégorielles.
             # Si processed_data.csv est utilisé dans train(), il est transformé via get_dummies.
             # On ne peut pas facilement reproduire le get_dummies ici sans re-implémenter la logique de train().
             # Simplification: Background de zéros pour SHAP Linear
             background = pd.DataFrame(0, index=np.arange(10), columns=model_columns)
             
             # Si pipeline, on transforme le background
             if hasattr(model, 'named_steps'):
                 for name, step in model.steps[:-1]:
                     if hasattr(step, 'transform'):
                         background = step.transform(background)
        else:
             background = pd.DataFrame(0, index=np.arange(10), columns=model_columns)
             if hasattr(model, 'named_steps'):
                 for name, step in model.steps[:-1]:
                     if hasattr(step, 'transform'):
                         background = step.transform(background)
    except:
        background = pd.DataFrame(0, index=np.arange(10), columns=model_columns)

    if class_name in ['RandomForestClassifier', 'XGBClassifier', 'GradientBoostingClassifier']:
        explainer = shap.TreeExplainer(estimator)
        # TreeExplainer gère tout seul ou nécessite data selon le modèle (sklearn RF a besoin que d'interne souvent, XGB aussi)
        shap_values_obj = explainer(explainer_input) # Nouvelle API
        shap_values = shap_values_obj.values
        
    elif class_name in ['LogisticRegression']:
        # LinearExplainer a besoin d'un masker (background data)
        explainer = shap.LinearExplainer(estimator, background)
        shap_values_obj = explainer(explainer_input)
        shap_values = shap_values_obj.values
        
    else:
        # Fallback (Naive Bayes, etc.) -> KernelExplainer (lent mais générique)
        # Ou on skip si c'est trop compliqué
        print(f"Modèle {class_name} : Utilisation de KernelExplainer (peut être lent)...")
        try:
             # KernelExplainer a besoin de la fonction de prédiction de proba
             # Si Pipeline, attention: estimator.predict_proba attend input scalé
             # Si on passe estimator.predict_proba, SHAP passera des inputs perturbés "scalés" (basé sur background scalé)
             predict_fn = estimator.predict_proba
             explainer = shap.KernelExplainer(predict_fn, background)
             shap_values = explainer.shap_values(explainer_input)
             # Kernel retourne souvent une liste pour classification
        except Exception as e:
             print(f"Impossible de calculer SHAP pour ce modèle : {e}")
             shap_values = None

    if shap_values is not None:
        # Standardisation des dimensions de shap_values
        vals = None
        
        # Si c'est un objet Explanation (nouvelle API)
        if shap_values_obj is not None and hasattr(shap_values_obj, "shape"): # C'est un objet Explanation ou array
             if len(shap_values.shape) == 2: # (n_samples, n_features) -> cas XGBoost binaire output margin parfois, ou LogReg
                  vals = shap_values[0]
             elif len(shap_values.shape) == 3: # (n_samples, n_features, n_classes)
                  vals = shap_values[0, :, 1] # Classe 1
        
        # Si c'est l'ancienne API (liste de arrays)
        if vals is None and isinstance(shap_values, list):
            if len(shap_values) > 1:
                vals = shap_values[1][0] # Classe 1
            else:
                 vals = shap_values[0][0] # Cas rare
        elif vals is None and isinstance(shap_values, np.ndarray):
             if len(shap_values.shape) == 2:
                  vals = shap_values[0]
        
        # Affichage
        if vals is not None:
            feature_importance = pd.DataFrame(list(zip(model_columns, vals)), columns=['Feature', 'SHAP_Value'])
            feature_importance['Abs_Value'] = feature_importance['SHAP_Value'].abs()
            feature_importance = feature_importance.sort_values(by='Abs_Value', ascending=False).head(7)
            
            print("\n--- POURQUOI CE RÉSULTAT ? (Top 7 Facteurs) ---")
            print("Une valeur positive (+) augmente le risque de départ.")
            print("Une valeur négative (-) réduit le risque (fidélise).")
            print("-" * 60)
            
            for index, row in feature_importance.iterrows():
                feature = row['Feature']
                shap_val = row['SHAP_Value']
                user_val = input_df[feature].values[0]
                
                direction = "Augmente le risque 🔴" if shap_val > 0 else "Réduit le risque 🟢"
                print(f"{feature:<25} : {user_val:>10.2f}  | Impact: {shap_val:>6.2f} ({direction})")
            print("-" * 60)
        else:
             print("Format SHAP non reconnu, impossible d'afficher les détails.")
    else:
        print("Pas d'explication détaillée disponible pour ce modèle.")

if __name__ == "__main__":
    predict_and_explain()
