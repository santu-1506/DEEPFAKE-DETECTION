import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import time
from datetime import datetime

# Model architecture (same as before)
class DeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(DeepfakeDetector, self).__init__()
        
        from torchvision.models import efficientnet_b0
        self.backbone = efficientnet_b0(pretrained=pretrained)
        
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def collect_all_images():
    """Collect all images including new modern AI ones"""
    
    print("📊 Collecting training data...")
    
    # Paths
    real_path = Path("data/processed/real")
    fake_path = Path("data/processed/fake")
    
    if not real_path.exists() or not fake_path.exists():
        print("❌ Processed data not found. Run data_preprocessing.py first!")
        return None, None, None, None
    
    # Collect images
    real_images = list(real_path.glob("*.jpg"))
    fake_images = list(fake_path.glob("*.jpg"))
    
    print(f"   📸 Real images: {len(real_images)}")
    print(f"   🤖 Fake/AI images: {len(fake_images)}")
    
    # Create paths and labels
    all_paths = real_images + fake_images
    all_labels = [0] * len(real_images) + [1] * len(fake_images)  # 0=real, 1=fake
    
    # Split into train/val (80/20)
    from sklearn.model_selection import train_test_split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    
    print(f"   📚 Training samples: {len(train_paths)}")
    print(f"   🔍 Validation samples: {len(val_paths)}")
    
    return train_paths, val_paths, train_labels, val_labels

def create_data_loaders(train_paths, val_paths, train_labels, val_labels, batch_size=16):
    """Create data loaders for training"""
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = ImageDataset(train_paths, train_labels, train_transform)
    val_dataset = ImageDataset(val_paths, val_labels, val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def train_model_with_modern_data():
    """Train the model with the updated dataset including modern AI images"""
    
    print("🚀 Starting Model Retraining with Modern AI Data")
    print("=" * 60)
    
    # Check if we have modern AI images
    modern_ai_path = Path("data/raw/modern_ai")
    if modern_ai_path.exists():
        ai_images = list(modern_ai_path.glob("*.jpg")) + list(modern_ai_path.glob("*.jpeg"))
        ai_images = [img for img in ai_images if not img.name.startswith('example_')]
        print(f"🤖 Found {len(ai_images)} modern AI images to add")
        
        if len(ai_images) > 0:
            # Copy modern AI images to fake dataset
            print("📋 Adding modern AI images to training dataset...")
            from data_collection_modern import copy_images_to_fake_dataset
            copy_images_to_fake_dataset(modern_ai_path)
            
            # Reprocess data
            print("🔄 Reprocessing data with new images...")
            os.system("python src/data_preprocessing.py")
    
    # Collect training data
    train_paths, val_paths, train_labels, val_labels = collect_all_images()
    if train_paths is None:
        return
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_paths, val_paths, train_labels, val_labels
    )
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    
    model = DeepfakeDetector(num_classes=2, pretrained=True)
    model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Training loop
    num_epochs = 15
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    print(f"🏋️ Training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_predictions = []
        val_true_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                
                val_predictions.extend(predicted.cpu().numpy())
                val_true_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        val_acc = accuracy_score(val_true_labels, val_predictions)
        avg_loss = running_loss / len(train_loader)
        
        train_losses.append(avg_loss)
        val_accuracies.append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}: Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            # Save model
            model_save_path = "models/trained/retrained_deepfake_detector.pth"
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_acc': best_val_acc,
                'train_losses': train_losses,
                'val_accuracies': val_accuracies
            }, model_save_path)
            
            print(f"   ✅ New best model saved! Accuracy: {best_val_acc:.4f}")
        
        scheduler.step()
    
    # Final evaluation
    print("\n📊 Final Evaluation:")
    final_precision = precision_score(val_true_labels, val_predictions)
    final_recall = recall_score(val_true_labels, val_predictions)
    final_f1 = f1_score(val_true_labels, val_predictions)
    
    print(f"   🎯 Accuracy: {val_acc:.4f}")
    print(f"   🎯 Precision: {final_precision:.4f}")
    print(f"   🎯 Recall: {final_recall:.4f}")
    print(f"   🎯 F1-Score: {final_f1:.4f}")
    
    # Save results
    results = {
        'final_accuracy': float(val_acc),
        'final_precision': float(final_precision),
        'final_recall': float(final_recall),
        'final_f1': float(final_f1),
        'training_time': f"{time.time() - time.time():.2f} seconds",
        'model_path': model_save_path,
        'timestamp': datetime.now().isoformat()
    }
    
    with open("models/trained/retrained_evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Retraining completed!")
    print(f"📁 Model saved to: {model_save_path}")
    print(f"📊 Results saved to: models/trained/retrained_evaluation_results.json")
    
    return model, results

if __name__ == "__main__":
    train_model_with_modern_data() 