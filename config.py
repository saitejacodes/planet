import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR
MODELS_DIR = BASE_DIR / "models"
VISUALIZATIONS_DIR = BASE_DIR / "visualizations"
INSIGHTS_DIR = BASE_DIR / "insights"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
for directory in [MODELS_DIR, VISUALIZATIONS_DIR, INSIGHTS_DIR, VECTOR_STORE_DIR]:
    directory.mkdir(exist_ok=True)
RAW_DATA_FILE = [f for f in os.listdir(DATA_DIR) if f.startswith("PSCompPars") and f.endswith(".csv")][0]
RAW_DATA_PATH = DATA_DIR / RAW_DATA_FILE
CLEANED_DATA_PATH = DATA_DIR / "cleaned_exoplanets.csv"
PLANET_TYPE_MODEL_PATH = MODELS_DIR / "planet_type_model.pkl"
HABITABILITY_MODEL_PATH = MODELS_DIR / "habitability_model.pkl"
INSIGHTS_PATH = INSIGHTS_DIR / "insights.json"
VECTOR_INDEX_PATH = VECTOR_STORE_DIR / "vector_index.faiss"
METADATA_PATH = VECTOR_STORE_DIR / "metadata.json"
FEATURES = ["pl_bmasse", "pl_orbper", "pl_eqt", "st_teff", "st_rad"]
RADIUS_COL = "pl_rade"
PLANET_TYPES = {0: "Terrestrial", 1: "Super-Earth", 2: "Neptune-like", 3: "Gas Giant"}
EARTH_MASS = 1.0
EARTH_TEMP = 288.0
SUN_TEMP = 5778.0
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
