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

# Change this to whichever vision-capable model
# you decide to use for the experiment.
#MODEL_NAME = "google/gemma-4-31b-it:free"
#MODEL_NAME="google/gemma-4-26b-a4b-it:free"
MODEL_NAME="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"

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