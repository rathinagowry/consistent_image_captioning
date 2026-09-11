import os
from dotenv import load_dotenv

load_dotenv()

# =========================
# API CONFIGURATION
# =========================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY not found. "
        "Add it to your .env file."
    )

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# =========================
# MODEL CONFIGURATION
# =========================

#MODEL_NAME = "google/gemma-4-31b-it:free"
#MODEL_NAME="google/gemma-4-26b-a4b-it:free"
MODEL_NAME="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
# ============================================================
# SEMANTIC EXTRACTION CONFIGURATION
# ============================================================

EXTRACTION_MODEL = (
    "nvidia/nemotron-3-ultra-550b-a55b:free"
)

EXTRACTION_TEMPERATURE = 0.0

# Number of times the SAME caption will be
# passed through the semantic extractor.
EXTRACTION_RUNS = 3

# Maximum extraction attempts for one run
EXTRACTION_MAX_RETRIES = 3

# Output directory for extraction experiments
EXTRACTION_RESULT_DIR = (
    "experiments/extraction_results"
)

EXTRACTION_JSON_OUTPUT = (
    f"{EXTRACTION_RESULT_DIR}/extraction_results.json"
)

EXTRACTION_CSV_OUTPUT = (
    f"{EXTRACTION_RESULT_DIR}/extraction_results.csv"
)
EXTRACTION_MODE = "stability"
STABILITY_CAPTIONS = [
    ("test_001", 3),
    ("test_002", 1),
    ("test_003", 3),
    ("test_004", 3),
    ("test_005", 3),
]
# =========================
# EXPERIMENT CONFIGURATION
# =========================

PROMPT = "Describe this image in one sentence."

# Number of repeated generations for each image
NUM_RUNS = 10

# Initial baseline temperature
TEMPERATURE = 0.0


# =========================
# PATH CONFIGURATION
# =========================

IMAGE_DIR = "data/images"

RESULT_DIR = "experiments/baseline_results"
JSON_OUTPUT = (
    f"{RESULT_DIR}/baseline_results_t{TEMPERATURE}.json"
)

CSV_OUTPUT = (
    f"{RESULT_DIR}/baseline_results_t{TEMPERATURE}.csv"
)