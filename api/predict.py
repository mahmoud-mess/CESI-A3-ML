"""
Endpoint: POST /api/predict
Prediction d'attrition avec ML
"""

from http.server import BaseHTTPRequestHandler
import json
import os
import sys

# Configuration des chemins pour Vercel
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
MODEL_DIR = os.path.join(BASE_DIR, "models")
COLUMNS_PATH = os.path.join(MODEL_DIR, "columns.pkl")

# Ajouter le dossier parent au path
sys.path.insert(0, BASE_DIR)

import joblib
import pandas as pd
import numpy as np

# Configuration
API_KEY = os.environ.get("API_KEY", "ATTRITION_SECRET_KEY_2024")

# Cache pour les modeles
_models_cache = {}
_columns_cache = None

# Liste exacte des 41 colonnes du modele (from columns.pkl)
EXPECTED_COLUMNS = [
    'Age', 'DistanceFromHome', 'Education', 'JobLevel', 'MonthlyIncome',
    'NumCompaniesWorked', 'PercentSalaryHike', 'StockOptionLevel',
    'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany',
    'YearsSinceLastPromotion', 'YearsWithCurrManager', 'JobInvolvement',
    'PerformanceRating', 'EnvironmentSatisfaction', 'JobSatisfaction',
    'WorkLifeBalance', 'MeanWorkingHours', 'WorkDays', 'OverTimeFrequency',
    'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely',
    'Department_Research & Development', 'Department_Sales',
    'EducationField_Life Sciences', 'EducationField_Marketing',
    'EducationField_Medical', 'EducationField_Other',
    'EducationField_Technical Degree', 'Gender_Male',
    'JobRole_Human Resources', 'JobRole_Laboratory Technician',
    'JobRole_Manager', 'JobRole_Manufacturing Director',
    'JobRole_Research Director', 'JobRole_Research Scientist',
    'JobRole_Sales Executive', 'JobRole_Sales Representative',
    'MaritalStatus_Married', 'MaritalStatus_Single'
]


def load_models():
    """Charge les modeles en cache."""
    global _models_cache, _columns_cache
    
    if _columns_cache is None:
        try:
            if os.path.exists(COLUMNS_PATH):
                _columns_cache = joblib.load(COLUMNS_PATH)
                print(f"[OK] Colonnes chargees: {len(_columns_cache)} features")
            else:
                print(f"[WARN] columns.pkl non trouve, utilisation des colonnes par defaut")
                _columns_cache = EXPECTED_COLUMNS
        except Exception as e:
            print(f"[ERREUR] Chargement columns: {e}")
            _columns_cache = EXPECTED_COLUMNS
    
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
                    print(f"[OK] Modele '{key}' charge")
                except Exception as e:
                    print(f"[ERREUR] Chargement {key}: {e}")
    
    return _models_cache, _columns_cache


def prepare_dataframe(employee_data, columns):
    """Prepare le DataFrame avec les 41 colonnes exactes du modele."""
    # Creer DataFrame avec toutes les colonnes a 0
    df = pd.DataFrame(0.0, index=[0], columns=columns)
    
    # === COLONNES NUMERIQUES (21 colonnes) ===
    numeric_mapping = {
        # API name -> Model column name
        'age': 'Age',
        'distance_from_home': 'DistanceFromHome',
        'education': 'Education',
        'job_level': 'JobLevel',
        'monthly_income': 'MonthlyIncome',
        'num_companies_worked': 'NumCompaniesWorked',
        'percent_salary_hike': 'PercentSalaryHike',
        'stock_option_level': 'StockOptionLevel',
        'total_working_years': 'TotalWorkingYears',
        'training_times_last_year': 'TrainingTimesLastYear',
        'years_at_company': 'YearsAtCompany',
        'years_since_last_promotion': 'YearsSinceLastPromotion',
        'years_with_curr_manager': 'YearsWithCurrManager',
        'job_involvement': 'JobInvolvement',
        'performance_rating': 'PerformanceRating',
        'environment_satisfaction': 'EnvironmentSatisfaction',
        'job_satisfaction': 'JobSatisfaction',
        'work_life_balance': 'WorkLifeBalance',
        'mean_working_hours': 'MeanWorkingHours',
        'work_days': 'WorkDays',
        'overtime_frequency': 'OverTimeFrequency'
    }
    
    # Valeurs par defaut realistes
    defaults = {
        'age': 35,
        'distance_from_home': 10,
        'education': 3,
        'job_level': 2,
        'monthly_income': 50000,
        'num_companies_worked': 2,
        'percent_salary_hike': 15,
        'stock_option_level': 1,
        'total_working_years': 10,
        'training_times_last_year': 3,
        'years_at_company': 5,
        'years_since_last_promotion': 1,
        'years_with_curr_manager': 3,
        'job_involvement': 3,
        'performance_rating': 3,
        'environment_satisfaction': 3,
        'job_satisfaction': 3,
        'work_life_balance': 3,
        'mean_working_hours': 8.0,
        'work_days': 230,
        'overtime_frequency': 0.1
    }
    
    # Remplir les colonnes numeriques
    for api_name, model_col in numeric_mapping.items():
        if model_col in df.columns:
            value = employee_data.get(api_name, defaults.get(api_name, 0))
            df[model_col] = float(value)
    
    # === COLONNES CATEGORIELLES ONE-HOT (20 colonnes) ===
    
    # BusinessTravel (reference: Non-Travel = 0,0)
    business_travel = employee_data.get('business_travel', 'Travel_Rarely')
    if business_travel == 'Travel_Frequently':
        df['BusinessTravel_Travel_Frequently'] = 1
    elif business_travel == 'Travel_Rarely':
        df['BusinessTravel_Travel_Rarely'] = 1
    # Si Non-Travel, les deux restent a 0
    
    # Department (reference: Human Resources = 0,0)
    department = employee_data.get('department', 'Research & Development')
    if department == 'Research & Development':
        df['Department_Research & Development'] = 1
    elif department == 'Sales':
        df['Department_Sales'] = 1
    # Si Human Resources, les deux restent a 0
    
    # EducationField (reference: Human Resources = tous a 0)
    education_field = employee_data.get('education_field', 'Life Sciences')
    edu_field_mapping = {
        'Life Sciences': 'EducationField_Life Sciences',
        'Marketing': 'EducationField_Marketing',
        'Medical': 'EducationField_Medical',
        'Other': 'EducationField_Other',
        'Technical Degree': 'EducationField_Technical Degree'
    }
    if education_field in edu_field_mapping:
        col = edu_field_mapping[education_field]
        if col in df.columns:
            df[col] = 1
    
    # Gender (reference: Female = 0)
    gender = employee_data.get('gender', 'Male')
    if gender == 'Male':
        df['Gender_Male'] = 1
    
    # JobRole (reference: Healthcare Representative = tous a 0)
    job_role = employee_data.get('job_role', 'Sales Executive')
    job_role_mapping = {
        'Human Resources': 'JobRole_Human Resources',
        'Laboratory Technician': 'JobRole_Laboratory Technician',
        'Manager': 'JobRole_Manager',
        'Manufacturing Director': 'JobRole_Manufacturing Director',
        'Research Director': 'JobRole_Research Director',
        'Research Scientist': 'JobRole_Research Scientist',
        'Sales Executive': 'JobRole_Sales Executive',
        'Sales Representative': 'JobRole_Sales Representative'
    }
    if job_role in job_role_mapping:
        col = job_role_mapping[job_role]
        if col in df.columns:
            df[col] = 1
    
    # MaritalStatus (reference: Divorced = 0,0)
    marital_status = employee_data.get('marital_status', 'Married')
    if marital_status == 'Married':
        df['MaritalStatus_Married'] = 1
    elif marital_status == 'Single':
        df['MaritalStatus_Single'] = 1
    
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
    """Recupere les top 5 facteurs importants."""
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
        elif hasattr(estimator, 'coef_'):
            coefs = estimator.coef_[0]
            feature_data = list(zip(columns, coefs, df.values[0]))
            feature_data.sort(key=lambda x: abs(x[1]), reverse=True)
            
            for feat, coef, val in feature_data[:5]:
                factors.append({
                    "feature": feat,
                    "value": float(val),
                    "impact": round(float(abs(coef)), 4),
                    "direction": "Augmente risque" if coef > 0 else "Reduit risque"
                })
    except Exception as e:
        print(f"[WARN] Feature importance: {e}")
    
    return factors


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        # Headers CORS
        self.send_header_cors = lambda: (
            self.send_header('Access-Control-Allow-Origin', '*'),
            self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS'),
            self.send_header('Access-Control-Allow-Headers', 'X-API-Key, Content-Type')
        )
        
        # Verifier la cle API
        api_key = self.headers.get('X-API-Key', '')
        
        if api_key != API_KEY:
            self.send_response(401)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                "error": "Cle API invalide",
                "hint": "Ajoutez le header X-API-Key"
            }).encode())
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
            
            if not models:
                raise Exception("Aucun modele charge")
            
            if model_key not in models:
                available = list(models.keys())
                raise Exception(f"Modele '{model_key}' non disponible. Utilisez: {available}")
            
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
                "success": True,
                "prediction": prediction,
                "probability": round(probability, 4),
                "risk_level": get_risk_level(probability),
                "model_used": model_key
            }
            
            # Facteurs d'importance
            if include_factors:
                response["factors"] = get_feature_importance(model, df, columns)
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except json.JSONDecodeError as e:
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                "success": False,
                "error": "JSON invalide",
                "detail": str(e)
            }).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                "success": False,
                "error": str(e),
                "debug": {
                    "model_dir": MODEL_DIR,
                    "model_dir_exists": os.path.exists(MODEL_DIR),
                    "columns_loaded": _columns_cache is not None,
                    "models_loaded": list(_models_cache.keys()) if _models_cache else []
                }
            }).encode())
        
        return
    
    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'X-API-Key, Content-Type')
        self.end_headers()
        return
    
    def do_GET(self):
        """Documentation de l'endpoint"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response = {
            "endpoint": "/api/predict",
            "method": "POST",
            "description": "Prediction du risque d'attrition d'un employe",
            "authentication": {
                "header": "X-API-Key",
                "required": True
            },
            "body": {
                "model": {
                    "type": "string",
                    "options": ["rf", "logreg", "nb"],
                    "default": "rf"
                },
                "employee": {
                    "type": "object",
                    "fields": {
                        "age": "int (18-70)",
                        "monthly_income": "float",
                        "years_at_company": "int",
                        "job_satisfaction": "int (1-4)",
                        "environment_satisfaction": "int (1-4)",
                        "work_life_balance": "int (1-4)",
                        "overtime_frequency": "float (0-1)",
                        "business_travel": "Non-Travel | Travel_Rarely | Travel_Frequently",
                        "department": "Human Resources | Research & Development | Sales",
                        "gender": "Female | Male",
                        "marital_status": "Divorced | Married | Single",
                        "job_role": "Healthcare Representative | Human Resources | ..."
                    }
                },
                "include_shap": {
                    "type": "boolean",
                    "default": True
                }
            },
            "example_request": {
                "model": "rf",
                "employee": {
                    "age": 30,
                    "monthly_income": 45000,
                    "years_at_company": 3,
                    "job_satisfaction": 2,
                    "environment_satisfaction": 2,
                    "overtime_frequency": 0.3,
                    "business_travel": "Travel_Frequently",
                    "marital_status": "Single"
                },
                "include_shap": True
            }
        }
        self.wfile.write(json.dumps(response, indent=2).encode())
        return
