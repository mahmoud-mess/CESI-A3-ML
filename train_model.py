import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

# --- CONFIGURATION ---
DATA_FILE = "processed_data.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "attrition_model.pkl")
ENCODERS_PATH = os.path.join(MODEL_DIR, "encoders.pkl")
COLUMNS_PATH = os.path.join(MODEL_DIR, "columns.pkl")

def get_model_choice():
    print("\n--- CHOIX DE L'ALGORITHME ---")
    print("1. Random Forest (Défaut)")
    print("2. Logistic Regression")
    print("3. XGBoost")
    print("4. Naive Bayes")
    
    choice = input("Votre choix (1-4) : ")
    
    if choice == "2":
        return "logreg", "Logistic Regression"
    elif choice == "3":
        return "xgboost", "XGBoost"
    elif choice == "4":
        return "nb", "Naive Bayes"
    else:
        return "rf", "Random Forest"

def train():
    print("--- DÉBUT DE L'ENTRAÎNEMENT (AVEC OPTIMISATION) ---")
    
    # Création du dossier models s'il n'existe pas
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # Choix du modèle
    model_key, model_name = get_model_choice()
    current_model_path = os.path.join(MODEL_DIR, f"attrition_model_{model_key}.pkl")
    print(f"Modèle sélectionné : {model_name}")
        
    # 1. Chargement
    print("Chargement du dataset...")
    df = pd.read_csv(DATA_FILE)
    
    # Suppression de l'ID (pas une feature prédictive)
    if 'EmployeeID' in df.columns:
        df.drop(columns=['EmployeeID'], inplace=True)
        
    # 2. Préparation des features (X) et cible (y)
    target_col = 'Attrition'
    
    # Encodage de la cible (Yes/No -> 1/0)
    le_target = LabelEncoder()
    df[target_col] = le_target.fit_transform(df[target_col])
    print(f"Distribution de la cible (0=No, 1=Yes) :\n{df[target_col].value_counts(normalize=True)}")
    
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    # 3. Encodage des variables catégorielles (One-Hot Encoding)
    X_encoded = pd.get_dummies(X, drop_first=True)
    final_columns = X_encoded.columns.tolist()
    
    # 4. Split Train/Test
    print("Séparation Train/Test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)
    
    # 5. Gestion du déséquilibre des classes (Class Weights)
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights_dict = dict(zip(classes, weights))
    
    # 6. Configuration de la GridSearch
    model = None
    param_grid = {}
    
    print(f"Configuration de la GridSearch pour {model_name}...")
    
    if model_key == "rf":
        model = RandomForestClassifier(class_weight=class_weights_dict, random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif model_key == "logreg":
        model = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42))
        # Attention à la syntaxe pipeline pour la grid search : stepname__param
        param_grid = {
            'logisticregression__C': [0.01, 0.1, 1, 10],
            'logisticregression__solver': ['lbfgs', 'liblinear']
        }
    elif model_key == "xgboost":
        count_neg = len(y_train[y_train == 0])
        count_pos = len(y_train[y_train == 1])
        scale_pos_weight = count_neg / count_pos
        
        model = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss')
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 1.0]
        }
    elif model_key == "nb":
        model = GaussianNB()
        param_grid = {
            'var_smoothing': np.logspace(0, -9, num=20)
        }
    
    # Lancement de la recherche
    print("Lancement de l'optimisation (cette étape peut prendre du temps)...")
    # On utilise 'f1' car on veut un bon compromis precision/recall, ou 'recall' si on veut absolument trouver les départs
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='f1', verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"\nMeilleurs paramètres trouvés : {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    
    # 7. Évaluation
    print("--- ÉVALUATION SUR LE TEST SET ---")
    y_pred = best_model.predict(X_test)
    
    # Gestion proba
    if hasattr(best_model, "predict_proba"):
        y_prob = best_model.predict_proba(X_test)[:, 1]
    else:
        # Cas SVM linear ou autre sans proba
        y_prob = [0]*len(y_test) 

    print("\nClassification Report :")
    print(classification_report(y_test, y_pred))
    
    print("\nMatrice de Confusion :")
    print(confusion_matrix(y_test, y_pred))
    
    try:
        auc = roc_auc_score(y_test, y_prob)
        print(f"\nAUC-ROC Score : {auc:.4f}")
    except:
        pass
    
    # Feature Importance (si disponible)
    estimator = best_model
    if hasattr(best_model, 'named_steps'):
        estimator = best_model.steps[-1][1]
        
    if hasattr(estimator, 'feature_importances_'):
        importances = pd.Series(estimator.feature_importances_, index=final_columns).sort_values(ascending=False)
        print("\nTop 5 Features les plus importantes :")
        print(importances.head(5))
    elif hasattr(estimator, 'coef_'):
        importances = pd.Series(estimator.coef_[0], index=final_columns).abs().sort_values(ascending=False)
        print("\nTop 5 Coefficients les plus importants (abs) :")
        print(importances.head(5))
    
    # 8. Sauvegarde des artefacts
    print("\nSauvegarde du modèle et des métadonnées...")
    joblib.dump(best_model, current_model_path)
    joblib.dump(final_columns, COLUMNS_PATH) 
    joblib.dump(le_target, ENCODERS_PATH)
    
    print(f"Modèle optimisé sauvegardé dans : {current_model_path}")
    print("--- ENTRAÎNEMENT TERMINÉ ---")

if __name__ == "__main__":
    train()
