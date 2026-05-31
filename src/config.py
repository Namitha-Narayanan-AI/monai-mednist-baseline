from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MEDNIST_DIR = DATA_DIR / "MedNIST"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
MODELS_DIR = RESULTS_DIR / "models"

RANDOM_SEED = 42
IMAGE_SIZE = 64
BATCH_SIZE = 64
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

for directory in [DATA_DIR, RESULTS_DIR, FIGURES_DIR, METRICS_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)