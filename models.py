
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from pathlib import Path

from .configs import OCT_CLASSES, FUNDUS_CLASSES, IMAGE_SIZE

class ModalityClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(ModalityClassifier, self).__init__()
        self.base_model = models.efficientnet_b0(pretrained=True)
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.base_model(x)

class OCTDiseaseClassifier(nn.Module):
    def __init__(self, num_classes=len(OCT_CLASSES)):
        super(OCTDiseaseClassifier, self).__init__()
        self.base_model = models.efficientnet_b0(pretrained=True)
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Linear(num_features, num_classes)
        self.classes = OCT_CLASSES
    
    def forward(self, x):
        return self.base_model(x)

class FundusDiseaseClassifier(nn.Module):
    def __init__(self, num_classes=len(FUNDUS_CLASSES)):
        super(FundusDiseaseClassifier, self).__init__()
        self.base_model = models.efficientnet_b0(pretrained=True)
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Linear(num_features, num_classes)
        self.classes = FUNDUS_CLASSES
    
    def forward(self, x):
        return self.base_model(x)

def create_modality_classifier(device='cpu', model_path=None):
    model = ModalityClassifier()
    if model_path and Path(model_path).exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def create_oct_classifier(device='cpu', model_path=None):
    model = OCTDiseaseClassifier()
    if model_path and Path(model_path).exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def create_fundus_classifier(device='cpu', model_path=None):
    model = FundusDiseaseClassifier()
    if model_path and Path(model_path).exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model