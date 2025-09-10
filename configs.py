import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = BASE_DIR / "data"
OCT_DATA_DIR = DATA_DIR / "oct"
FUNDUS_DATA_DIR = DATA_DIR / "fundus"

# Model paths
MODEL_DIR = BASE_DIR / "models"
OCT_MODEL_PATH = MODEL_DIR / "oct_model.pth"
FUNDUS_MODEL_PATH = MODEL_DIR / "fundus_model.pth"
MODALITY_MODEL_PATH = MODEL_DIR / "modality_model.pth"

# Create directories if they don't exist
for path in [DATA_DIR, OCT_DATA_DIR, FUNDUS_DATA_DIR, MODEL_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Dataset classes
# Based on the OCT dataset: https://www.kaggle.com/datasets/obulisainaren/retinal-oct-c8
OCT_CLASSES = [
    'CNV',           # Choroidal Neovascularization
    'DME',           # Diabetic Macular Edema
    'DRUSEN',        # Drusen
    'NORMAL',        # Normal retina
    'AMD',           # Age-related Macular Degeneration
    'RVO',           # Retinal Vein Occlusion
    'CSC',           # Central Serous Chorioretinopathy
    'ERM'            # Epiretinal Membrane
]

# Based on the fundus dataset: https://www.kaggle.com/datasets/kssanjaynithish03/retinal-fundus-images
FUNDUS_CLASSES = [
    'Diabetic Retinopathy',
    'Glaucoma',
    'Cataract',
    'Normal',
    'Hypertensive Retinopathy',
    'Myopia',
    'AMD',
    'Other'
]

# Training parameters
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
LEARNING_RATE = 0.001
NUM_EPOCHS = 50

# Inference parameters
CONFIDENCE_THRESHOLD = 0.7