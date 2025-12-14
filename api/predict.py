"""
Endpoint: POST /api/predict
Prediction d'attrition avec ML
"""

from http.server import BaseHTTPRequestHandler
import json
import os
import sys

# Ajouter le dossier parent au path pour importer les modeles
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
import numpy as np

# Configuration
API_KEY = os.environ.get("API_KEY", "ATTRITION_SECRET_KEY_2024")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
COLUMNS_PATH = os.path.join(MODEL_DIR, "columns.pkl")

# Cache pour les modeles (evite de recharger a chaque requete)
_models_cache = {}
_columns_cache = None


def load_models():
    """Charge les modeles en cache."""
    global _models_cache, _columns_cache
    
    if _columns_cache is None:
        try:
            _columns_cache = joblib.load(COLUMNS_PATH)
        except Exception as e:
            print(f"Erreur chargement columns: {e}")
            _columns_cache = []
    
    model_files = {
        "rf": "attrition_model_rf.pkl",
        "logreg": "attrition_model_logreg.pkl",
        "nb": "attrition_model_nb.pkl"
    }
    
    for key, filename in model_files.items():
        if key not in _models_cache:
            path = os.path.join(MODEL_DIR, filename)
            if os.path.exists(path):
                try:
                    _models_cache[key] = joblib.load(path)
                except Exception as e:
                    print(f"Erreur chargement {key}: {e}")
    
    return _models_cache, _columns_cache


def prepare_dataframe(employee_data, columns):
    """Prepare le DataFrame pour le modele."""
    df = pd.DataFrame(0, index=[0], columns=columns)
    
    # Mapping des noms
    numeric_mapping = {
        'age': 'Age',
        'monthly_income': 'MonthlyIncome',
        'total_working_years': 'TotalWorkingYears',
        'years_at_company': 'YearsAtCompany',
        'years_with_curr_manager': 'YearsWithCurrManager',
        'distance_from_home': 'DistanceFromHome',
        'environment_satisfaction': 'EnvironmentSatisfaction',
        'job_satisfaction': 'JobSatisfaction',
        'work_life_balance': 'WorkLifeBalance',
        'num_companies_worked': 'NumCompaniesWorked',
        'percent_salary_hike': 'PercentSalaryHike',
        'training_times_last_year': 'TrainingTimesLastYear',
        'years_since_last_promotion': 'YearsSinceLastPromotion',
        'stock_option_level': 'StockOptionLevel',
        'job_level': 'JobLevel',
        'education': 'Education',
        'job_involvement': 'JobInvolvement',
        'performance_rating': 'PerformanceRating',
        'mean_working_hours': 'MeanWorkingHours',
        'work_days': 'WorkDays',
        'overtime_frequency': 'OverTimeFrequency'
    }
    
    # Valeurs par defaut
    defaults = {
        'age': 35, 'monthly_income': 50000, 'total_working_years': 10,
        'years_at_company': 5, 'years_with_curr_manager': 3, 'distance_from_home': 10,
        'environment_satisfaction': 3, 'job_satisfaction': 3, 'work_life_balance': 3,
        'num_companies_worked': 2, 'percent_salary_hike': 15, 'training_times_last_year': 3,
        'years_since_last_promotion': 1, 'stock_option_level': 1, 'job_level': 2,
        'education': 3, 'job_involvement': 3, 'performance_rating': 3,
        'mean_working_hours': 8.0, 'work_days': 230, 'overtime_frequency': 0.1
    }
    
    # Remplir les numeriques
    for api_name, model_name in numeric_mapping.items():
        if model_name in df.columns:
            value = employee_data.get(api_name, defaults.get(api_name, 0))
            df[model_name] = value
    
    # Encoder les categorielles
    categorical_mapping = {
        'business_travel': 'BusinessTravel',
        'department': 'Department',
        'gender': 'Gender',
        'marital_status': 'MaritalStatus',
        'over_time': 'OverTime',
        'job_role': 'JobRole',
        'education_field': 'EducationField'
    }
    
    cat_defaults = {
        'business_travel': 'Travel_Rarely',
        'department': 'Research & Development',
        'gender': 'Male',
        'marital_status': 'Married',
        'over_time': 'No',
        'job_role': 'Sales Executive',
        'education_field': 'Life Sciences'
    }
    
    for api_name, prefix in categorical_mapping.items():
        value = employee_data.get(api_name, cat_defaults.get(api_name, ''))
        
        if api_name == 'gender' and value == 'Male':
            if 'Gender_Male' in df.columns:
                df['Gender_Male'] = 1
        elif api_name == 'over_time' and value == 'Yes':
            if 'OverTime_Yes' in df.columns:
                df['OverTime_Yes'] = 1
        else:
            col_name = f"{prefix}_{value}"
            if col_name in df.columns:
                df[col_name] = 1
    
    return df


def get_risk_level(probability):
    """Determine le niveau de risque."""
    if probability < 0.25:
        return "Faible"
    elif probability < 0.50:
        return "Modere"
    elif probability < 0.75:
        return "Eleve"
    else:
        return "Critique"


def get_feature_importance(model, df, columns):
    """Recupere les facteurs importants."""
    factors = []
    
    try:
        estimator = model
        if hasattr(model, 'named_steps'):
            estimator = model.steps[-1][1]
        
        if hasattr(estimator, 'feature_importances_'):
            importances = estimator.feature_importances_
            feature_data = list(zip(columns, importances, df.values[0]))
            feature_data.sort(key=lambda x: abs(x[1]), reverse=True)
            
            for feat, imp, val in feature_data[:5]:
                factors.append({
                    "feature": feat,
                    "value": float(val),
                    "impact": round(float(imp), 4),
                    "direction": "Facteur important" if imp > 0.05 else "Facteur mineur"
                })
    except Exception as e:
        print(f"Erreur feature importance: {e}")
    
    return factors


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        # Verifier la cle API
        api_key = self.headers.get('X-API-Key', '')
        
        if api_key != API_KEY:
            self.send_response(401)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Cle API invalide"}).encode())
            return
        
        try:
            # Lire le body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            
            # Extraire les donnees
            model_key = data.get('model', 'rf')
            employee_data = data.get('employee', {})
            include_factors = data.get('include_shap', True)
            
            # Charger les modeles
            models, columns = load_models()
            
            if not columns:
                raise Exception("Colonnes non chargees")
            
            if model_key not in models:
                raise Exception(f"Modele '{model_key}' non disponible. Disponibles: {list(models.keys())}")
            
            model = models[model_key]
            
            # Preparer les donnees
            df = prepare_dataframe(employee_data, columns)
            
            # Prediction
            if hasattr(model, "predict_proba"):
                probability = float(model.predict_proba(df)[0][1])
            else:
                probability = float(model.predict(df)[0])
            
            prediction = "Yes" if probability > 0.5 else "No"
            
            # Reponse
            response = {
                "prediction": prediction,
                "probability": round(probability, 4),
                "risk_level": get_risk_level(probability),
                "model_used": model_key
            }
            
            # Facteurs d'importance (optionnel)
            if include_factors:
                response["factors"] = get_feature_importance(model, df, columns)
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
        
        return
    
    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'X-API-Key, Content-Type')
        self.end_headers()
        return

