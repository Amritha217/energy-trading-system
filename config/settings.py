"""

Centralised configuration for the project.
 
All parameters are here so changes propagate automatically to every
module. Import constants directly:
 
    from config.settings import FEATURES, THRESHOLD
 
Environment variables (via .env file)
--------------------------------------
HF_API_TOKEN : Hugging Face API token for the Inference API (optional).
               Without this, the agent falls back to the rule-based engine.
"""


from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv()

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "data" / "models"

for d in [DATA_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Hugging Face
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_MODEL_ID  = "mistralai/Mistral-7B-Instruct-v0.2"   

# Data
TARGET_COL = "load"
DATE_COL   = "Datetime"

# Model
TRAIN_SPLIT      = 0.8
THRESHOLD        = 0.001
TRANSACTION_COST = 0.0005
INITIAL_CAPITAL  = 10_000

FEATURES = [
    "hour", "dayofweek", "month", "quarter",
    "lag_24", "lag_168",
    "rolling_mean_24", "rolling_std_24",
    "volatility_24"
]