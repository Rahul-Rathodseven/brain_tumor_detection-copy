import os

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_DIR = os.path.join(BASE_DIR, "data", "train")
TEST_DIR  = os.path.join(BASE_DIR, "data", "test")

MODEL_DIR        = os.path.join(BASE_DIR, "models")
BEST_MODEL_PATH  = os.path.join(MODEL_DIR, "brain_tumor_vgg16.keras")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "final_model.keras")

OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR   = os.path.join(OUTPUT_DIR, "plots")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

for _dir in [MODEL_DIR, PLOTS_DIR, REPORTS_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ── Hyperparameters ──────────────────────────────────────────────────────────
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
RANDOM_SEED   = 42

# ── CLASSES: MUST match the alphabetical order of Kaggle training folders ────
#
# Kaggle dataset folder names (what the model was trained on):
#   glioma / meningioma / notumor / pituitary
#
# flow_from_directory sorts alphabetically → indices assigned as:
#   0 = glioma
#   1 = meningioma
#   2 = notumor      ← lowercase, no underscore
#   3 = pituitary
#
# Your LOCAL data folders are named differently:
#   glioma / meningioma / No_Tumor / pituitary
# But that does NOT matter here — CLASSES must match what the MODEL learned,
# not what your local folders are called.
#
CLASSES     = ["glioma", "meningioma", "pituitary", "notumor"]
#CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASSES)

EPOCHS        = 50
LEARNING_RATE = 1e-4
FINE_TUNE_LR  = 1e-5
PATIENCE      = 5