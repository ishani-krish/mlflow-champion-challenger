"""
config.py - the ONE place for settings. Edit STUDENT_USN, nothing else (unless you know why).
All other scripts import from here so names always match.
"""
from pathlib import Path

# ---- paths (all relative to this folder, so it works on any machine) ----
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
EVAL_CSV = DATA_DIR / "eval_inputs.csv"
NEW_DATA_CSV = DATA_DIR / "new_data.csv"

# ---- you ----
STUDENT_USN = "1MS25CI060"          

# ---- MLflow names ----
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "hf-sentiment-champion-challenger"
REGISTERED_MODEL_NAME = "sentiment-model"
TASK = "sentiment"
DATASET_NAME = "glue/sst2 validation subset"
SEED = 42
MAX_LENGTH = 128

# ---- the two models ----
MODELS = [
    {"slot": "A", "role": "champion",
     "hf_model_id": "distilbert-base-uncased-finetuned-sst-2-english",
     "short": "distilbert-sst2"},
    {"slot": "B", "role": "challenger",
     "hf_model_id": "siebert/sentiment-roberta-large-english",
     "short": "roberta-large"},
]

# ======================================================================
# CHAMPION RULE  -- written BEFORE the final run. DO NOT CHANGE AFTER
# SEEING RESULTS (-2 marks). It must match the rule in README.md.
# ======================================================================
PRIMARY_METRIC = "macro_f1"     # higher is better
MARGIN = 0.02                   # Challenger must beat Champion by >= 2 percentage points
LATENCY_GUARDRAIL = 10.0        # reject Challenger if avg latency > 10x Champion's
# Tie-breaker: if |difference| < MARGIN, the faster model (lower avg latency) wins.
