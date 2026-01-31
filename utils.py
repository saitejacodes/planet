import joblib
import json
import numpy as np
import config
def save_model(model, filepath):
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")
def load_model(filepath):
    return joblib.load(filepath)
def save_json(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to {filepath}")
def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)
def calculate_habitability_score(row):
    temp_score = 0
    if not np.isnan(row['pl_eqt']):
        temp_score = np.exp(-((row['pl_eqt'] - config.EARTH_TEMP)**2) / (2 * 50**2))
    mass_score = 0
    if not np.isnan(row['pl_bmasse']):
        mass_score = np.exp(-((np.log10(row['pl_bmasse']) - np.log10(config.EARTH_MASS))**2) / (2 * 0.5**2))
    star_score = 0
    if not np.isnan(row['st_teff']):
        star_score = np.exp(-((row['st_teff'] - config.SUN_TEMP)**2) / (2 * 1000**2))
    total_score = (0.5 * temp_score) + (0.3 * mass_score) + (0.2 * star_score)
    return total_score
def get_planet_type(radius):
    if radius < 1.25:
        return 0
    elif radius < 2.0:
        return 1
    elif radius < 6.0:
        return 2
    else:
        return 3
