import cv2
import numpy as np
from pathlib import Path
from src.inference import DeepfakeInference

def test_image_loading():
    """Test basic image loading functionality"""
    
    print("🔍 Testing Image Loading...")
    
    # Test a single validation image
    test_image = Path("data/splits/val/real/real_0001.jpg")
    
    if not test_image.exists():
        print(f"❌ Test image not found: {test_image}")
        return False
    
    print(f"📁 Testing image: {test_image}")
    
    # Test OpenCV loading
    try:
        image = cv2.imread(str(test_image))
        if image is None:
            print("❌ OpenCV failed to load image")
            return False
        else:
            print(f"✅ OpenCV loaded image: {image.shape}")
    except Exception as e:
        print(f"❌ OpenCV error: {e}")
        return False
    
    # Test RGB conversion
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"✅ RGB conversion successful: {image_rgb.shape}")
    except Exception as e:
        print(f"❌ RGB conversion error: {e}")
        return False
    
    return True

def test_model_loading():
    """Test model loading"""
    
    print("\n🧠 Testing Model Loading...")
    
    try:
        detector = DeepfakeInference()
        print("✅ Model loaded successfully")
        return detector
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return None

def test_single_prediction():
    """Test prediction on a single image"""
    
    print("\n🎯 Testing Single Prediction...")
    
    # First test image loading
    if not test_image_loading():
        return
    
    # Then test model loading
    detector = test_model_loading()
    if detector is None:
        return
    
    # Test prediction
    test_image = Path("data/splits/val/real/real_0001.jpg")
    
    try:
        print(f"🔍 Making prediction on: {test_image.name}")
        result = detector.predict_single_image(str(test_image))
        
        print("✅ Prediction successful!")
        print(f"   Prediction: {result['prediction']}")
        print(f"   Confidence: {result['confidence']:.1f}%")
        print(f"   Real prob: {result['real_probability']:.1f}%")
        print(f"   Fake prob: {result['fake_probability']:.1f}%")
        print(f"   Time: {result['inference_time_ms']:.1f}ms")
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_prediction() 