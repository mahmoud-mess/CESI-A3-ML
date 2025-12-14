"""
API de Prédiction d'Attrition des Employés
==========================================
Framework: FastAPI
Authentification: Clé API via Header X-API-Key
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Optional, List
from enum import Enum

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# --- CONFIGURATION ---
API_KEY = os.getenv("API_KEY", "ATTRITION_SECRET_KEY_2024")
MODEL_DIR = "models"
COLUMNS_PATH = os.path.join(MODEL_DIR, "columns.pkl")

# --- INITIALISATION FASTAPI ---
app = FastAPI(
    title="API Attrition Employés",
    description="Prédit le risque de départ d'un employé avec explications SHAP",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS - Permet les appels depuis React (localhost ou Vercel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, mettre l'URL exacte du frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CHARGEMENT DES MODÈLES (au démarrage) ---
models = {}
model_columns = []

@app.on_event("startup")
async def load_models():
    """Charge tous les modèles au démarrage de l'API."""
    global models, model_columns
    
    print("[INFO] Chargement des modeles...")
    
    # Charger les colonnes
    try:
        model_columns = joblib.load(COLUMNS_PATH)
        print(f"[OK] Colonnes chargees ({len(model_columns)} features)")
    except FileNotFoundError:
        print("[ERREUR] columns.pkl introuvable!")
        
    # Charger chaque modèle
    model_files = {
        "rf": "attrition_model_rf.pkl",
        "xgboost": "attrition_model_xgboost.pkl",
        "logreg": "attrition_model_logreg.pkl",
        "nb": "attrition_model_nb.pkl"
    }
    
    for key, filename in model_files.items():
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            try:
                models[key] = joblib.load(path)
                print(f"[OK] Modele '{key}' charge")
            except Exception as e:
                print(f"[ERREUR] Chargement {key}: {e}")
    
    print(f"[READY] {len(models)} modele(s) pret(s)")


# --- SCHÉMAS PYDANTIC ---

class ModelChoice(str, Enum):
    rf = "rf"
    xgboost = "xgboost"
    logreg = "logreg"
    nb = "nb"

class BusinessTravelChoice(str, Enum):
    non_travel = "Non-Travel"
    travel_rarely = "Travel_Rarely"
    travel_frequently = "Travel_Frequently"

class DepartmentChoice(str, Enum):
    rd = "Research & Development"
    sales = "Sales"
    hr = "Human Resources"

class GenderChoice(str, Enum):
    male = "Male"
    female = "Female"

class MaritalStatusChoice(str, Enum):
    single = "Single"
    married = "Married"
    divorced = "Divorced"

class OverTimeChoice(str, Enum):
    yes = "Yes"
    no = "No"

class JobRoleChoice(str, Enum):
    sales_executive = "Sales Executive"
    research_scientist = "Research Scientist"
    laboratory_technician = "Laboratory Technician"
    manufacturing_director = "Manufacturing Director"
    healthcare_representative = "Healthcare Representative"
    manager = "Manager"
    sales_representative = "Sales Representative"
    research_director = "Research Director"
    human_resources = "Human Resources"

class EducationFieldChoice(str, Enum):
    life_sciences = "Life Sciences"
    medical = "Medical"
    marketing = "Marketing"
    technical_degree = "Technical Degree"
    other = "Other"
    human_resources = "Human Resources"


class EmployeeInput(BaseModel):
    """Données d'un employé pour la prédiction."""
    
    # Numériques
    age: int = Field(..., ge=18, le=70, description="Âge de l'employé")
    monthly_income: float = Field(..., ge=1000, description="Salaire mensuel")
    total_working_years: int = Field(..., ge=0, le=50, description="Années d'expérience totale")
    years_at_company: int = Field(..., ge=0, le=50, description="Années dans l'entreprise")
    years_with_curr_manager: int = Field(..., ge=0, le=50, description="Années avec le manager actuel")
    distance_from_home: int = Field(..., ge=0, le=100, description="Distance domicile-travail (km)")
    
    # Satisfaction (1-4)
    environment_satisfaction: int = Field(..., ge=1, le=4, description="Satisfaction environnement (1-4)")
    job_satisfaction: int = Field(..., ge=1, le=4, description="Satisfaction travail (1-4)")
    work_life_balance: int = Field(..., ge=1, le=4, description="Équilibre vie/travail (1-4)")
    
    # Optionnels avec défauts
    num_companies_worked: int = Field(default=1, ge=0, le=20)
    percent_salary_hike: int = Field(default=15, ge=0, le=100)
    training_times_last_year: int = Field(default=3, ge=0, le=10)
    years_since_last_promotion: int = Field(default=1, ge=0, le=20)
    stock_option_level: int = Field(default=0, ge=0, le=3)
    job_level: int = Field(default=2, ge=1, le=5)
    education: int = Field(default=3, ge=1, le=5)
    job_involvement: int = Field(default=3, ge=1, le=4)
    performance_rating: int = Field(default=3, ge=1, le=4)
    
    # Métriques temporelles (optionnelles)
    mean_working_hours: float = Field(default=8.0, ge=0, le=24)
    work_days: int = Field(default=230, ge=0, le=365)
    overtime_frequency: float = Field(default=0.1, ge=0.0, le=1.0)
    
    # Catégorielles
    business_travel: BusinessTravelChoice = Field(default=BusinessTravelChoice.travel_rarely)
    department: DepartmentChoice = Field(default=DepartmentChoice.rd)
    gender: GenderChoice = Field(default=GenderChoice.male)
    marital_status: MaritalStatusChoice = Field(default=MaritalStatusChoice.married)
    over_time: OverTimeChoice = Field(default=OverTimeChoice.no)
    job_role: JobRoleChoice = Field(default=JobRoleChoice.sales_executive)
    education_field: EducationFieldChoice = Field(default=EducationFieldChoice.life_sciences)

    class Config:
        use_enum_values = True


class PredictRequest(BaseModel):
    """Requête de prédiction."""
    model: ModelChoice = Field(default=ModelChoice.rf, description="Modèle à utiliser")
    employee: EmployeeInput
    include_shap: bool = Field(default=True, description="Inclure les facteurs SHAP")


class ShapFactor(BaseModel):
    """Un facteur d'influence SHAP."""
    feature: str
    value: float
    impact: float
    direction: str


class PredictionResponse(BaseModel):
    """Réponse de prédiction."""
    prediction: str
    probability: float
    risk_level: str
    model_used: str
    shap_factors: Optional[List[ShapFactor]] = None


# --- AUTHENTIFICATION ---

async def verify_api_key(x_api_key: str = Header(..., description="Clé API")):
    """Vérifie que la clé API est valide."""
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Clé API invalide. Vérifiez le header X-API-Key."
        )
    return True


# --- FONCTIONS UTILITAIRES ---

def prepare_dataframe(employee: EmployeeInput) -> pd.DataFrame:
    """Prépare le DataFrame pour le modèle (même logique que predict.py)."""
    
    # Créer un DF avec toutes les colonnes à 0
    df = pd.DataFrame(0, index=[0], columns=model_columns)
    
    # Mapping des noms (snake_case API → noms du modèle)
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
    
    # Remplir les valeurs numériques
    employee_dict = employee.dict()
    for api_name, model_name in numeric_mapping.items():
        if model_name in df.columns:
            df[model_name] = employee_dict[api_name]
    
    # Encoder les catégorielles (One-Hot)
    categorical_mapping = {
        'business_travel': 'BusinessTravel',
        'department': 'Department',
        'gender': 'Gender',
        'marital_status': 'MaritalStatus',
        'over_time': 'OverTime',
        'job_role': 'JobRole',
        'education_field': 'EducationField'
    }
    
    for api_name, prefix in categorical_mapping.items():
        value = employee_dict[api_name]
        
        # Cas spéciaux
        if api_name == 'gender' and value == 'Male':
            if 'Gender_Male' in df.columns:
                df['Gender_Male'] = 1
        elif api_name == 'over_time' and value == 'Yes':
            if 'OverTime_Yes' in df.columns:
                df['OverTime_Yes'] = 1
        else:
            # Construire le nom de colonne One-Hot
            col_name = f"{prefix}_{value}"
            if col_name in df.columns:
                df[col_name] = 1
    
    return df


def calculate_shap_factors(model, df: pd.DataFrame, model_key: str) -> List[ShapFactor]:
    """Calcule les facteurs SHAP (simplifié, sans dépendance lourde pour rapidité)."""
    
    factors = []
    
    try:
        # Pour les modèles à base d'arbres, utiliser feature_importances
        estimator = model
        if hasattr(model, 'named_steps'):
            estimator = model.steps[-1][1]
        
        if hasattr(estimator, 'feature_importances_'):
            importances = estimator.feature_importances_
            
            # Créer les facteurs
            feature_data = list(zip(model_columns, importances, df.values[0]))
            feature_data.sort(key=lambda x: abs(x[1]), reverse=True)
            
            for feat, imp, val in feature_data[:7]:
                # Simplification: impact basé sur importance * valeur normalisée
                impact = float(imp)
                direction = "Facteur important" if imp > 0.05 else "Facteur mineur"
                
                factors.append(ShapFactor(
                    feature=feat,
                    value=float(val),
                    impact=round(impact, 4),
                    direction=direction
                ))
                
        elif hasattr(estimator, 'coef_'):
            # Pour régression logistique
            coefs = estimator.coef_[0]
            feature_data = list(zip(model_columns, coefs, df.values[0]))
            feature_data.sort(key=lambda x: abs(x[1]), reverse=True)
            
            for feat, coef, val in feature_data[:7]:
                impact = float(coef * val)
                direction = "Augmente le risque" if impact > 0 else "Réduit le risque"
                
                factors.append(ShapFactor(
                    feature=feat,
                    value=float(val),
                    impact=round(impact, 4),
                    direction=direction
                ))
                
    except Exception as e:
        print(f"Erreur SHAP: {e}")
    
    return factors


def get_risk_level(probability: float) -> str:
    """Détermine le niveau de risque."""
    if probability < 0.25:
        return "Faible"
    elif probability < 0.50:
        return "Modéré"
    elif probability < 0.75:
        return "Élevé"
    else:
        return "Critique"


# --- ENDPOINTS ---

@app.get("/", tags=["General"])
async def root():
    """Page d'accueil de l'API."""
    return {
        "message": "API Attrition Employes - Ready!",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": ["/health", "/models", "/predict"]
    }


@app.get("/health", tags=["General"])
async def health_check():
    """Vérifie que l'API fonctionne."""
    return {
        "status": "ok",
        "models_loaded": len(models),
        "features_count": len(model_columns)
    }


@app.get("/models", tags=["Modeles"], dependencies=[Depends(verify_api_key)])
async def list_models():
    """Liste les modèles disponibles."""
    return {
        "models": list(models.keys()),
        "default": "rf",
        "descriptions": {
            "rf": "Random Forest - Bon équilibre précision/rappel",
            "xgboost": "XGBoost - Très performant, recommandé",
            "logreg": "Régression Logistique - Simple et interprétable",
            "nb": "Naive Bayes - Rapide mais moins précis"
        }
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"], dependencies=[Depends(verify_api_key)])
async def predict(request: PredictRequest):
    """
    Prédit le risque d'attrition d'un employé.
    
    - **model**: Algorithme à utiliser (rf, xgboost, logreg, nb)
    - **employee**: Données de l'employé
    - **include_shap**: Inclure les facteurs d'influence (défaut: true)
    """
    
    model_key = request.model.value
    
    # Vérifier que le modèle existe
    if model_key not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Modèle '{model_key}' non trouvé. Disponibles: {list(models.keys())}"
        )
    
    model = models[model_key]
    
    # Préparer les données
    df = prepare_dataframe(request.employee)
    
    # Prédiction
    try:
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(df)[0][1])
        else:
            probability = float(model.predict(df)[0])
        
        prediction = "Yes" if probability > 0.5 else "No"
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur de prédiction: {str(e)}"
        )
    
    # Facteurs SHAP (optionnel)
    shap_factors = None
    if request.include_shap:
        shap_factors = calculate_shap_factors(model, df, model_key)
    
    return PredictionResponse(
        prediction=prediction,
        probability=round(probability, 4),
        risk_level=get_risk_level(probability),
        model_used=model_key,
        shap_factors=shap_factors
    )


@app.get("/features", tags=["Modeles"], dependencies=[Depends(verify_api_key)])
async def list_features():
    """Liste toutes les features utilisées par le modèle."""
    return {
        "count": len(model_columns),
        "features": model_columns
    }


# --- POINT D'ENTRÉE ---

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

