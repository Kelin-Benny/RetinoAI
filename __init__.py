"""
RetinoAI Pro - AI-Powered Retinal Disease Diagnosis System
"""

__version__ = "0.1.0"
__author__ = "RetinoAI Team"
__license__ = "MIT"

from .configs import (
    BASE_DIR, DATA_DIR, OCT_DATA_DIR, FUNDUS_DATA_DIR,
    MODEL_DIR, OCT_MODEL_PATH, FUNDUS_MODEL_PATH, MODALITY_MODEL_PATH,
    OCT_CLASSES, FUNDUS_CLASSES, BATCH_SIZE, IMAGE_SIZE,
    LEARNING_RATE, NUM_EPOCHS, CONFIDENCE_THRESHOLD
)

from .datasets import RetinalDataset, get_oct_datasets, get_fundus_datasets, get_data_loaders
from .models import (
    ModalityClassifier, OCTDiseaseClassifier, FundusDiseaseClassifier,
    create_modality_classifier, create_oct_classifier, create_fundus_classifier
)
from .train import train_oct_model, train_fundus_model
from .inference import RetinoAIInference
from .gradcam import GradCAM, overlay_heatmap, get_grad_cam
from .report import generate_pdf_report
from .app import create_app

# Export main classes and functions
__all__ = [
    # Configs
    'BASE_DIR', 'DATA_DIR', 'OCT_DATA_DIR', 'FUNDUS_DATA_DIR',
    'MODEL_DIR', 'OCT_MODEL_PATH', 'FUNDUS_MODEL_PATH', 'MODALITY_MODEL_PATH',
    'OCT_CLASSES', 'FUNDUS_CLASSES', 'BATCH_SIZE', 'IMAGE_SIZE',
    'LEARNING_RATE', 'NUM_EPOCHS', 'CONFIDENCE_THRESHOLD',
    
    # Datasets
    'RetinalDataset', 'get_oct_datasets', 'get_fundus_datasets', 'get_data_loaders',
    
    # Models
    'ModalityClassifier', 'OCTDiseaseClassifier', 'FundusDiseaseClassifier',
    'create_modality_classifier', 'create_oct_classifier', 'create_fundus_classifier',
    
    # Training
    'train_model', 'train_oct_model', 'train_fundus_model', 'train_modality_classifier',
    
    # Inference
    'RetinoAIInference',
    
    # Grad-CAM
    'GradCAM', 'overlay_heatmap', 'get_grad_cam',
    
    # Reporting
    'generate_pdf_report',
    
    # App
    'create_app'
]