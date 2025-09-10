import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path

from .models import create_modality_classifier, create_oct_classifier, create_fundus_classifier
from .configs import OCT_CLASSES, FUNDUS_CLASSES, IMAGE_SIZE, MODALITY_MODEL_PATH, OCT_MODEL_PATH, FUNDUS_MODEL_PATH

class RetinoAIInference:
    def __init__(self, device='cpu'):
        self.device = device
        self.modality_classifier = create_modality_classifier(device, MODALITY_MODEL_PATH)
        self.oct_classifier = create_oct_classifier(device, OCT_MODEL_PATH)
        self.fundus_classifier = create_fundus_classifier(device, FUNDUS_MODEL_PATH)
        
        self.modality_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.oct_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.fundus_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def predict_modality(self, image):
        """Predict whether image is OCT or fundus"""
        image_tensor = self.modality_transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.modality_classifier(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][pred_class].item()
        
        modality = "OCT" if pred_class == 0 else "Fundus"
        return modality, confidence
    
    def predict_oct(self, image):
        """Predict OCT disease classification"""
        image_tensor = self.oct_transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.oct_classifier(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][pred_class].item()
            all_probabilities = probabilities.cpu().numpy()[0]
        
        diagnosis = OCT_CLASSES[pred_class]
        return diagnosis, confidence, all_probabilities
    
    def predict_fundus(self, image):
        """Predict fundus disease classification"""
        image_tensor = self.fundus_transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.fundus_classifier(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][pred_class].item()
            all_probabilities = probabilities.cpu().numpy()[0]
        
        diagnosis = FUNDUS_CLASSES[pred_class]
        return diagnosis, confidence, all_probabilities
    
    def predict(self, image, patient_data=None):
        """Complete prediction pipeline"""
        # Step 1: Determine modality
        modality, modality_confidence = self.predict_modality(image)
        
        # Step 2: Route to appropriate model
        if modality == "OCT":
            diagnosis, confidence, probabilities = self.predict_oct(image)
            class_names = OCT_CLASSES
        else:
            diagnosis, confidence, probabilities = self.predict_fundus(image)
            class_names = FUNDUS_CLASSES
        
        # Prepare results
        results = {
            'modality': modality,
            'modality_confidence': round(modality_confidence * 100, 2),
            'diagnosis': diagnosis,
            'confidence': round(confidence * 100, 2),
            'probabilities': {cls: round(prob * 100, 2) for cls, prob in zip(class_names, probabilities)},
            'patient_data': patient_data or {}
        }
        
        return results
    
    def batch_predict(self, image_paths, patient_data_list=None):
        """Batch prediction for multiple images"""
        results = []
        
        for i, image_path in enumerate(image_paths):
            try:
                image = Image.open(image_path).convert('RGB')
                patient_data = patient_data_list[i] if patient_data_list and i < len(patient_data_list) else None
                result = self.predict(image, patient_data)
                result['filename'] = Path(image_path).name
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                results.append({
                    'filename': Path(image_path).name,
                    'error': str(e)
                })
        
        return results