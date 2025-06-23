import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from PIL import Image
import time

class DeepfakeDetector(nn.Module):
    """
    Same model architecture as training - needed to load weights
    """
    
    def __init__(self, num_classes=2, pretrained=False):
        super(DeepfakeDetector, self).__init__()
        
        from torchvision.models import efficientnet_b0
        self.backbone = efficientnet_b0(pretrained=pretrained)
        
        # Same architecture as training
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

class DeepfakeInference:
    """
    Main class for deepfake detection inference
    
    What this does:
    - Loads your trained AI model
    - Preprocesses new images
    - Makes predictions with confidence scores
    - Provides interpretable results
    """
    
    def __init__(self, model_path="models/trained/best_deepfake_detector.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(model_path)
        
        # Load the trained model
        print(f"🔄 Loading trained model from {model_path}...")
        self.model = self._load_model()
        
        # Image preprocessing (same as training)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✅ Model loaded successfully on {self.device}")
    
    def _load_model(self):
        """Load the trained model weights"""
        
        # Initialize model
        model = DeepfakeDetector(num_classes=2, pretrained=False)
        
        # Load saved weights
        if self.model_path.exists():
            checkpoint = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"📊 Model trained to {checkpoint.get('val_acc', 'unknown')}% validation accuracy")
        else:
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        model.to(self.device)
        model.eval()  # Set to evaluation mode
        return model
    
    def preprocess_image(self, image_path):
        """
        Preprocess a single image for inference
        
        Steps:
        1. Load image
        2. Convert to RGB
        3. Resize to 224x224
        4. Normalize like training data
        5. Add batch dimension
        """
        
        # Load image
        if isinstance(image_path, str):
            image_path = str(image_path)  # Ensure string path
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
        else:
            image = image_path  # Already loaded image
        
        # Ensure image is valid numpy array
        if not isinstance(image, np.ndarray):
            raise ValueError(f"Invalid image data type: {type(image)}")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply same transforms as training
        image_tensor = self.transform(image_rgb)
        
        # Add batch dimension: (3, 224, 224) -> (1, 3, 224, 224)
        image_batch = image_tensor.unsqueeze(0)
        
        return image_batch.to(self.device), image_rgb
    
    def predict_single_image(self, image_path, return_confidence=True):
        """
        Predict if a single image is real or fake
        
        Returns:
        - prediction: 'real' or 'fake'
        - confidence: probability score (0-100%)
        - raw_scores: detailed model outputs
        """
        
        start_time = time.time()
        
        # Preprocess image
        image_tensor, original_image = self.preprocess_image(image_path)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence_score = probabilities[0][predicted_class].item()
        
        # Convert to human-readable results
        class_names = ['real', 'fake']
        prediction = class_names[predicted_class]
        confidence_percentage = confidence_score * 100
        
        # Detailed scores
        real_score = probabilities[0][0].item() * 100
        fake_score = probabilities[0][1].item() * 100
        
        inference_time = time.time() - start_time
        
        result = {
            'prediction': prediction,
            'confidence': confidence_percentage,
            'real_probability': real_score,
            'fake_probability': fake_score,
            'inference_time_ms': inference_time * 1000,
            'image_path': str(image_path)
        }
        
        if return_confidence:
            return result
        else:
            return prediction
    
    def predict_batch(self, image_paths):
        """
        Predict multiple images at once (more efficient)
        """
        
        results = []
        print(f"🔍 Processing {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths, 1):
            try:
                result = self.predict_single_image(image_path)
                results.append(result)
                print(f"   {i}/{len(image_paths)}: {Path(image_path).name} -> {result['prediction']} ({result['confidence']:.1f}%)")
            except Exception as e:
                print(f"   ❌ Error processing {image_path}: {e}")
                results.append({
                    'image_path': str(image_path),
                    'error': str(e)
                })
        
        return results
    
    def visualize_prediction(self, image_path, save_path=None):
        """
        Create a visualization showing the image and prediction
        """
        
        # Get prediction
        result = self.predict_single_image(image_path)
        
        # Load and display image
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        
        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(image_rgb)
        plt.title(f"Original Image\n{Path(image_path).name}")
        plt.axis('off')
        
        # Prediction results
        plt.subplot(1, 2, 2)
        
        # Color based on prediction
        color = 'green' if result['prediction'] == 'real' else 'red'
        
        # Bar chart of probabilities
        categories = ['Real', 'Fake']
        probabilities = [result['real_probability'], result['fake_probability']]
        colors = ['green' if result['prediction'] == 'real' else 'lightgray',
                 'red' if result['prediction'] == 'fake' else 'lightgray']
        
        bars = plt.bar(categories, probabilities, color=colors)
        plt.ylim(0, 100)
        plt.ylabel('Confidence (%)')
        plt.title(f"Prediction: {result['prediction'].upper()}\nConfidence: {result['confidence']:.1f}%")
        
        # Add percentage labels on bars
        for bar, prob in zip(bars, probabilities):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{prob:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Visualization saved to: {save_path}")
        
        plt.show()
        
        return result

def test_on_validation_samples():
    """
    Test the inference system on some validation images
    """
    
    print("🧪 Testing Inference System...")
    print("=" * 50)
    
    # Initialize inference
    detector = DeepfakeInference()
    
    # Get some validation images to test
    val_real_dir = Path("data/splits/val/real")
    val_fake_dir = Path("data/splits/val/fake")
    
    test_images = []
    
    # Add a few real images
    if val_real_dir.exists():
        real_images = list(val_real_dir.glob("*.jpg"))[:3]
        test_images.extend(real_images)
    
    # Add a few fake images
    if val_fake_dir.exists():
        fake_images = list(val_fake_dir.glob("*.jpg"))[:3]
        test_images.extend(fake_images)
    
    if not test_images:
        print("❌ No validation images found for testing")
        return
    
    # Test batch prediction
    results = detector.predict_batch(test_images)
    
    # Show detailed results
    print(f"\n📊 Detailed Results:")
    print("-" * 30)
    
    correct_predictions = 0
    total_predictions = 0
    
    for result in results:
        if 'error' in result:
            continue
            
        image_path = Path(result['image_path'])
        true_label = 'real' if 'real' in str(image_path) else 'fake'
        predicted_label = result['prediction']
        
        is_correct = true_label == predicted_label
        correct_predictions += is_correct
        total_predictions += 1
        
        status = "✅" if is_correct else "❌"
        print(f"{status} {image_path.name}")
        print(f"    True: {true_label}, Predicted: {predicted_label}")
        print(f"    Confidence: {result['confidence']:.1f}%")
        print(f"    Time: {result['inference_time_ms']:.1f}ms")
        print()
    
    accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
    print(f"🎯 Test Accuracy: {accuracy:.1f}% ({correct_predictions}/{total_predictions})")
    
    # Create a visualization for the first image
    if test_images:
        print(f"\n📊 Creating visualization for: {test_images[0].name}")
        detector.visualize_prediction(test_images[0], save_path="docs/inference_example.png")

def predict_custom_image(image_path):
    """
    Predict a custom image provided by user
    """
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"🔍 Analyzing image: {Path(image_path).name}")
    print("-" * 40)
    
    # Initialize detector
    detector = DeepfakeInference()
    
    # Make prediction
    result = detector.predict_single_image(image_path)
    
    # Display results
    print(f"📊 Results:")
    print(f"   Prediction: {result['prediction'].upper()}")
    print(f"   Confidence: {result['confidence']:.1f}%")
    print(f"   Real probability: {result['real_probability']:.1f}%")
    print(f"   Fake probability: {result['fake_probability']:.1f}%")
    print(f"   Processing time: {result['inference_time_ms']:.1f}ms")
    
    # Create visualization
    detector.visualize_prediction(image_path, save_path="docs/custom_prediction.png")
    
    return result

if __name__ == "__main__":
    # Test the inference system
    test_on_validation_samples()
    
    # Example of predicting a custom image
    # Uncomment and provide your own image path:
    # predict_custom_image("path/to/your/image.jpg")
