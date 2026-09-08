from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "respiratory_disease_hospitalization_forecasting_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"

RANDOM_STATE = 42
TARGET = "respiratory_hospitalizations"
TEST_SIZE = 0.20
FORECAST_HORIZON = 30
