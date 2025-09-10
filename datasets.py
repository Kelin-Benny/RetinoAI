import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path

from .configs import OCT_DATA_DIR, FUNDUS_DATA_DIR, OCT_CLASSES, FUNDUS_CLASSES, IMAGE_SIZE, BATCH_SIZE

class RetinalDataset(Dataset):
    def __init__(self, data_dir, classes, transform=None, mode='train'):
        self.data_dir = data_dir
        self.classes = classes
        self.transform = transform
        self.mode = mode
        self.images = []
        self.labels = []
        
        # Load images and labels
        for class_idx, class_name in enumerate(classes):
            class_dir = data_dir / mode / class_name
            if class_dir.exists():
                for img_file in class_dir.iterdir():
                    if img_file.suffix.lower() in ['.jpeg', '.jpg', '.png', '.bmp']:
                        self.images.append(str(img_file))
                        self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_oct_datasets():
    """Get OCT train, validation, and test datasets"""
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = RetinalDataset(OCT_DATA_DIR, OCT_CLASSES, transform, 'train')
    val_dataset = RetinalDataset(OCT_DATA_DIR, OCT_CLASSES, transform, 'val')
    test_dataset = RetinalDataset(OCT_DATA_DIR, OCT_CLASSES, transform, 'test')
    
    return train_dataset, val_dataset, test_dataset

def get_fundus_datasets():
    """Get fundus train, validation, and test datasets"""
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = RetinalDataset(FUNDUS_DATA_DIR, FUNDUS_CLASSES, transform, 'train')
    val_dataset = RetinalDataset(FUNDUS_DATA_DIR, FUNDUS_CLASSES, transform, 'val')
    test_dataset = RetinalDataset(FUNDUS_DATA_DIR, FUNDUS_CLASSES, transform, 'test')
    
    return train_dataset, val_dataset, test_dataset

def get_data_loaders():
    """Get data loaders for both OCT and fundus datasets"""
    oct_train, oct_val, oct_test = get_oct_datasets()
    fundus_train, fundus_val, fundus_test = get_fundus_datasets()
    
    oct_train_loader = DataLoader(oct_train, batch_size=BATCH_SIZE, shuffle=True)
    oct_val_loader = DataLoader(oct_val, batch_size=BATCH_SIZE, shuffle=False)
    oct_test_loader = DataLoader(oct_test, batch_size=BATCH_SIZE, shuffle=False)
    
    fundus_train_loader = DataLoader(fundus_train, batch_size=BATCH_SIZE, shuffle=True)
    fundus_val_loader = DataLoader(fundus_val, batch_size=BATCH_SIZE, shuffle=False)
    fundus_test_loader = DataLoader(fundus_test, batch_size=BATCH_SIZE, shuffle=False)
    
    return {
        'oct': {
            'train': oct_train_loader,
            'val': oct_val_loader,
            'test': oct_test_loader
        },
        'fundus': {
            'train': fundus_train_loader,
            'val': fundus_val_loader,
            'test': fundus_test_loader
        }
    }