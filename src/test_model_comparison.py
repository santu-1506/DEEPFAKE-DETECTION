import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
from pathlib import Path
import time

# Model architecture
class DeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
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

def load_model(model_path):
    """Load a trained model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepfakeDetector(num_classes=2, pretrained=False)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, device

def predict_image(model, image_path, device):
    """Make prediction on a single image"""
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence_score = probabilities[0][predicted_class].item()
    
    class_names = ['real', 'fake']
    prediction = class_names[predicted_class]
    confidence_percentage = confidence_score * 100
    
    real_score = probabilities[0][0].item() * 100
    fake_score = probabilities[0][1].item() * 100
    
    return {
        'prediction': prediction,
        'confidence': confidence_percentage,
        'real_probability': real_score,
        'fake_probability': fake_score
    }

def test_model_comparison():
    """Compare old vs new model performance on modern AI images"""
    
    print("🔬 Model Comparison Test")
    print("=" * 50)
    
    # Model paths
    old_model_path = "models/trained/best_deepfake_detector.pth"
    new_model_path = "models/trained/retrained_deepfake_detector.pth"
    
    # Test images (modern AI images)
    test_images_path = Path("data/raw/modern_ai")
    
    if not test_images_path.exists():
        print("❌ No modern AI images found for testing")
        return
    
    # Get test images
    test_images = list(test_images_path.glob("*.jpg")) + list(test_images_path.glob("*.jpeg"))
    test_images = [img for img in test_images if not img.name.startswith('example_')]
    
    if len(test_images) == 0:
        print("❌ No test images found")
        return
    
    print(f"🖼️ Testing on {len(test_images)} modern AI images")
    
    # Load models
    try:
        print("\n📊 Loading original model...")
        old_model, device = load_model(old_model_path)
        print("✅ Original model loaded")
    except Exception as e:
        print(f"❌ Could not load original model: {e}")
        return
    
    try:
        print("📊 Loading retrained model...")
        new_model, device = load_model(new_model_path)
        print("✅ Retrained model loaded")
    except Exception as e:
        print(f"❌ Could not load retrained model: {e}")
        print("⏳ Model may still be training...")
        return
    
    # Test both models
    print(f"\n🔍 Testing models on modern AI images...")
    print("-" * 70)
    print(f"{'Image':<25} {'Original':<15} {'Retrained':<15} {'Improvement'}")
    print("-" * 70)
    
    old_correct = 0
    new_correct = 0
    total_tests = len(test_images)
    
    for img_path in test_images:
        try:
            # Test original model
            old_result = predict_image(old_model, img_path, device)
            
            # Test retrained model  
            new_result = predict_image(new_model, img_path, device)
            
            # Check if predictions are correct (should be 'fake' for AI images)
            old_is_correct = old_result['prediction'] == 'fake'
            new_is_correct = new_result['prediction'] == 'fake'
            
            if old_is_correct:
                old_correct += 1
            if new_is_correct:
                new_correct += 1
            
            # Format results
            old_pred = f"{old_result['prediction']} ({old_result['confidence']:.1f}%)"
            new_pred = f"{new_result['prediction']} ({new_result['confidence']:.1f}%)"
            
            improvement = "✅ Fixed!" if (not old_is_correct and new_is_correct) else "📈 Better" if new_result['fake_probability'] > old_result['fake_probability'] else "➡️ Same"
            
            print(f"{img_path.name:<25} {old_pred:<15} {new_pred:<15} {improvement}")
            
        except Exception as e:
            print(f"{img_path.name:<25} ERROR: {str(e)}")
    
    # Summary
    print("-" * 70)
    print(f"\n📊 Final Results:")
    print(f"   Original Model Accuracy: {old_correct}/{total_tests} ({old_correct/total_tests*100:.1f}%)")
    print(f"   Retrained Model Accuracy: {new_correct}/{total_tests} ({new_correct/total_tests*100:.1f}%)")
    
    improvement = new_correct - old_correct
    if improvement > 0:
        print(f"   🎉 Improvement: +{improvement} correct predictions!")
        print(f"   📈 Accuracy increased by {improvement/total_tests*100:.1f} percentage points")
    elif improvement == 0:
        print(f"   ➡️ No change in accuracy")
    else:
        print(f"   📉 Performance decreased by {abs(improvement)} predictions")
    
    # Save results
    results = {
        'original_accuracy': old_correct / total_tests,
        'retrained_accuracy': new_correct / total_tests,
        'improvement': improvement,
        'total_tests': total_tests,
        'test_images': [str(img) for img in test_images]
    }
    
    with open("models/trained/comparison_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: models/trained/comparison_results.json")

if __name__ == "__main__":
    test_model_comparison() 