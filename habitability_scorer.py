import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import config
import utils
import os
def train_habitability_model():
    print("Training Habitability Scorer...")
    if not os.path.exists(config.CLEANED_DATA_PATH):
        print("Cleaned data not found. Please run data_ingestion.py first.")
        return
    df = pd.read_csv(config.CLEANED_DATA_PATH)
    print("Calculating habitability scores...")
    df['habitability_score'] = df.apply(utils.calculate_habitability_score, axis=1)
    X_cols = config.FEATURES
    y_col = 'habitability_score'
    X = df[X_cols]
    y = df[y_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model Performance:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R²: {r2:.6f}")
    utils.save_model(model, config.HABITABILITY_MODEL_PATH)
    importances = pd.DataFrame({
        'feature': X_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nFeature Importances:")
    print(importances)
train_habitability_model()
