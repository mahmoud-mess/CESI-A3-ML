import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

# --- CONFIGURATION ---
DATA_FILE = "processed_data.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "attrition_model.pkl")
ENCODERS_PATH = os.path.join(MODEL_DIR, "encoders.pkl")
COLUMNS_PATH = os.path.join(MODEL_DIR, "columns.pkl")

def train():
    print("--- DÉBUT DE L'ENTRAÎNEMENT ---")
    
    # Création du dossier models s'il n'existe pas
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
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
    # On utilise pd.get_dummies qui est plus simple pour gérer les colonnes
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # Sauvegarde de la liste des colonnes finales pour l'inférence
    final_columns = X_encoded.columns.tolist()
    
    # 4. Split Train/Test
    print("Séparation Train/Test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)
    
    # 5. Gestion du déséquilibre des classes (Class Weights)
    # On calcule les poids pour pénaliser les erreurs sur la classe minoritaire (départs)
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))
    print(f"Poids des classes calculés : {class_weights}")
    
    # 6. Entraînement du modèle
    # On utilise RandomForest qui est robuste et gère bien les class_weight
    print("Entraînement du RandomForest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight=class_weights,
        random_state=42,
        n_jobs=-1
    )
    
    # Alternative : GradientBoosting (souvent meilleur mais plus long à tuner)
    # model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    # Note: Sklearn GBDT ne supporte pas nativement class_weight aussi facilement que RF dans les vieilles versions
    
    model.fit(X_train, y_train)
    
    # 7. Évaluation
    print("--- ÉVALUATION SUR LE TEST SET ---")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\nClassification Report :")
    print(classification_report(y_test, y_pred))
    
    print("\nMatrice de Confusion :")
    print(confusion_matrix(y_test, y_pred))
    
    auc = roc_auc_score(y_test, y_prob)
    print(f"\nAUC-ROC Score : {auc:.4f}")
    
    # Feature Importance
    importances = pd.Series(model.feature_importances_, index=final_columns).sort_values(ascending=False)
    print("\nTop 5 Features les plus importantes :")
    print(importances.head(5))
    
    # 8. Sauvegarde des artefacts
    print("\nSauvegarde du modèle et des métadonnées...")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(final_columns, COLUMNS_PATH)
    # On n'a pas utilisé de LabelEncoder complexe pour X, mais si on en avait, il faudrait les sauver
    # On sauve juste le target encoder au cas où
    joblib.dump(le_target, ENCODERS_PATH)
    
    print(f"Modèle sauvegardé dans : {MODEL_PATH}")
    print("--- ENTRAÎNEMENT TERMINÉ ---")

if __name__ == "__main__":
    train()
