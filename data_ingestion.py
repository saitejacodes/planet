import pandas as pd
import numpy as np
import config
import utils
def load_and_preprocess():
    print(f"Loading data from {config.RAW_DATA_PATH}...")
    df = pd.read_csv(config.RAW_DATA_PATH, comment='#')
    print(f"Initial shape: {df.shape}")
    keep_cols = config.FEATURES + [config.RADIUS_COL, "pl_name", "discoverymethod"]
    df = df[keep_cols].copy()
    df.dropna(subset=keep_cols, inplace=True)
    print(f"Shape after dropping nulls: {df.shape}")
    print("Generating planet type labels...")
    df['y'] = df[config.RADIUS_COL].apply(utils.get_planet_type)
    df['planet_type'] = df['y'].map(config.PLANET_TYPES)
    print(f"Saving processed data to {config.CLEANED_DATA_PATH}...")
    df.to_csv(config.CLEANED_DATA_PATH, index=False)
    print("Data ingestion complete.")
    print("Class distribution:")
    print(df['planet_type'].value_counts())
    return df
load_and_preprocess()
