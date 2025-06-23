import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import cv2
import numpy as np
from pathlib import Path
import json
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class DeepfakeDataset(Dataset):
    """
    Custom dataset class for loading deepfake detection data
    
    What this does:
    - Loads images from organized folders (real/fake)
    - Applies data transformations (augmentation)
    - Returns images with labels (0=real, 1=fake)
    """
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        
        # Load real images (label = 0)
        real_dir = self.data_dir / "real"
        if real_dir.exists():
            for img_path in real_dir.glob("*.jpg"):
                self.samples.append((str(img_path), 0))  # 0 = real
        
        # Load fake images (label = 1)
        fake_dir = self.data_dir / "fake"
        if fake_dir.exists():
            for img_path in fake_dir.glob("*.jpg"):
                self.samples.append((str(img_path), 1))  # 1 = fake
        
        print(f"📁 Loaded {len(self.samples)} samples from {data_dir}")
        
        # Count by class
        real_count = sum(1 for _, label in self.samples if label == 0)
        fake_count = sum(1 for _, label in self.samples if label == 1)
        print(f"   Real: {real_count}, Fake: {fake_count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_data_transforms():
    """
    Create data augmentation and normalization transforms
    
    Why data augmentation?
    - Increases dataset size artificially
    - Makes model more robust to variations
    - Prevents overfitting
    
    Transforms explained:
    - RandomHorizontalFlip: Mirror images sometimes
    - RandomRotation: Slight rotations
    - ColorJitter: Change brightness/contrast slightly
    - Normalize: Standard ImageNet normalization
    """
    
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),  # 50% chance to flip
        transforms.RandomRotation(degrees=10),    # Rotate up to 10 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet standards
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transforms (no augmentation, just normalize)
    val_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_transforms

class DeepfakeDetector(nn.Module):
    """
    Our AI model for deepfake detection
    
    Architecture choice: EfficientNet-B0
    Why EfficientNet?
    - State-of-the-art accuracy
    - Efficient (good speed/accuracy balance)
    - Pre-trained on ImageNet (transfer learning)
    - Great for image classification tasks
    """
    
    def __init__(self, num_classes=2, pretrained=True):
        super(DeepfakeDetector, self).__init__()
        
        # Load pre-trained EfficientNet
        from torchvision.models import efficientnet_b0
        self.backbone = efficientnet_b0(pretrained=pretrained)
        
        # Replace the final classifier layer
        # EfficientNet-B0 has 1280 features before the final layer
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),  # Prevent overfitting
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)  # 2 classes: real, fake
        )
    
    def forward(self, x):
        return self.backbone(x)

def create_data_loaders(data_path="data/splits", batch_size=16):
    """
    Create PyTorch data loaders for training and validation
    
    Batch size explained:
    - 16 images processed together
    - Good balance for GPU memory
    - Stable gradient updates
    """
    
    train_transforms, val_transforms = get_data_transforms()
    
    # Create datasets
    train_dataset = DeepfakeDataset(
        data_dir=Path(data_path) / "train",
        transform=train_transforms
    )
    
    val_dataset = DeepfakeDataset(
        data_dir=Path(data_path) / "val", 
        transform=val_transforms
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,      # Randomize order each epoch
        num_workers=0,     # No multiprocessing on Windows
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,     # No need to shuffle validation
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    """
    Train the deepfake detection model
    
    Training process:
    1. Show model training images
    2. Model makes predictions
    3. Calculate how wrong the predictions are (loss)
    4. Adjust model weights to be more accurate
    5. Repeat for all images, many times (epochs)
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Training on: {device}")
    
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Good for classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler (reduce LR when plateau)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_path = "models/trained/best_deepfake_detector.pth"
    
    # Create models directory
    Path("models/trained").mkdir(parents=True, exist_ok=True)
    
    print(f"\n🚀 Starting training for {num_epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        print(f"\n📚 Epoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Progress update
            if (batch_idx + 1) % 5 == 0:
                print(f"   Batch {batch_idx+1}/{len(train_loader)}: Loss = {loss.item():.4f}")
        
        # Calculate training metrics
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc
            }, best_model_path)
            print(f"✅ New best model saved! Validation accuracy: {val_acc:.2f}%")
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch results
        epoch_time = time.time() - epoch_start_time
        print(f"📊 Epoch {epoch+1} Results ({epoch_time:.1f}s):")
        print(f"   Train: Loss = {train_loss:.4f}, Accuracy = {train_acc:.2f}%")
        print(f"   Val:   Loss = {val_loss:.4f}, Accuracy = {val_acc:.2f}%")
        print(f"   Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    print(f"\n🎉 Training complete!")
    print(f"💎 Best validation accuracy: {best_val_acc:.2f}%")
    print(f"💾 Best model saved to: {best_model_path}")
    
    return model, history

def evaluate_model(model, val_loader):
    """
    Comprehensive evaluation of the trained model
    
    Metrics calculated:
    - Accuracy: Overall correctness
    - Precision: When predicting fake, how often correct?
    - Recall: Of all real fakes, how many did we catch?
    - F1-Score: Balance between precision and recall
    - Confusion Matrix: Detailed breakdown of predictions
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    print("🔍 Evaluating model performance...")
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    print(f"\n📊 Final Model Performance:")
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.title('Deepfake Detection - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save confusion matrix
    cm_path = "docs/confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f"📈 Confusion matrix saved to: {cm_path}")
    plt.show()
    
    # Save evaluation results
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist()
    }
    
    results_path = "models/trained/evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 Evaluation results saved to: {results_path}")
    
    return results

def plot_training_history(history):
    """
    Create visualizations of the training process
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Training Loss', color='blue')
    ax1.plot(history['val_loss'], label='Validation Loss', color='red')
    ax1.set_title('Model Loss During Training')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Training Accuracy', color='blue')
    ax2.plot(history['val_acc'], label='Validation Accuracy', color='red')
    ax2.set_title('Model Accuracy During Training')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save training history plot
    history_path = "docs/training_history.png"
    plt.savefig(history_path, dpi=150, bbox_inches='tight')
    print(f"📈 Training history saved to: {history_path}")
    plt.show()

def main_training():
    """
    Main training pipeline
    """
    
    print("🚀 Starting AI Model Training Pipeline...")
    print("=" * 60)
    
    # Check for CUDA
    if torch.cuda.is_available():
        print(f"🎮 GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("💻 Training on CPU (GPU recommended for faster training)")
    
    # Step 1: Create data loaders
    print("\n📋 Step 1: Loading Data")
    train_loader, val_loader = create_data_loaders()
    
    # Step 2: Initialize model
    print("\n📋 Step 2: Initializing AI Model")
    model = DeepfakeDetector(num_classes=2, pretrained=True)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Step 3: Train model
    print("\n📋 Step 3: Training Model")
    model, history = train_model(model, train_loader, val_loader, num_epochs=15)
    
    # Step 4: Evaluate model
    print("\n📋 Step 4: Model Evaluation")
    results = evaluate_model(model, val_loader)
    
    # Step 5: Save training plots
    print("\n📋 Step 5: Saving Visualizations")
    plot_training_history(history)
    
    print("\n" + "=" * 60)
    print("🎉 AI Model Training Complete!")
    print("\n📋 What you now have:")
    print("   ✅ Trained deepfake detection AI model")
    print("   ✅ Model performance evaluation")
    print("   ✅ Training history visualizations")
    print("   ✅ Ready for inference and deployment")
    
    if results['accuracy'] > 0.85:
        print(f"\n🌟 Excellent! Your model achieved {results['accuracy']*100:.1f}% accuracy!")
        print("🚀 Ready for the next step: Building the API!")
    elif results['accuracy'] > 0.75:
        print(f"\n👍 Good! Your model achieved {results['accuracy']*100:.1f}% accuracy!")
        print("💡 Consider adding more data or training longer for better results.")
    else:
        print(f"\n⚠️  Model accuracy is {results['accuracy']*100:.1f}%. Consider:")
        print("   - Adding more training data")
        print("   - Training for more epochs") 
        print("   - Adjusting hyperparameters")

if __name__ == "__main__":
    main_training()
