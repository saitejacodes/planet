import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
import config
import utils
import os
def train_classifier():
    print("Training Planet Type Classifier...")
    if not os.path.exists(config.CLEANED_DATA_PATH):
        print("Cleaned data not found. Please run data_ingestion.py first.")
        return
    df = pd.read_csv(config.CLEANED_DATA_PATH)
    X_cols = [col for col in config.FEATURES if col != config.RADIUS_COL]
    y_col = 'y'
    X = df[X_cols]
    y = df[y_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    }
    best_model = None
    best_score = 0
    best_name = ""
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, target_names=list(config.PLANET_TYPES.values())))
        if acc > best_score:
            best_score = acc
            best_model = model
            best_name = name
    print(f"Best model: {best_name} with accuracy {best_score:.4f}")
    utils.save_model(best_model, config.PLANET_TYPE_MODEL_PATH)
    if hasattr(best_model, 'feature_importances_'):
        importances = pd.DataFrame({
            'feature': X_cols,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\nFeature Importances:")
        print(importances)
train_classifier()
