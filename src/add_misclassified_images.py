import os
import shutil
from pathlib import Path
import time

def add_misclassified_images():
    """
    Helper script to add misclassified AI images to the training dataset
    """
    
    print("🔄 Adding Misclassified AI Images to Training Dataset")
    print("=" * 60)
    
    # Source and destination paths
    modern_ai_path = Path("data/raw/modern_ai")
    fake_training_path = Path("data/raw/fake")
    
    # Create directories if they don't exist
    modern_ai_path.mkdir(parents=True, exist_ok=True)
    fake_training_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Source: {modern_ai_path}")
    print(f"📁 Destination: {fake_training_path}")
    
    # Find all images in modern_ai directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
    modern_ai_images = []
    
    for ext in image_extensions:
        modern_ai_images.extend(modern_ai_path.glob(f"*{ext}"))
        modern_ai_images.extend(modern_ai_path.glob(f"*{ext.upper()}"))
    
    # Filter out existing examples
    modern_ai_images = [img for img in modern_ai_images 
                       if not img.name.startswith('example_') 
                       and not img.name.startswith('README')]
    
    if len(modern_ai_images) == 0:
        print("📋 Instructions:")
        print("   1. Save AI images that were misclassified as 'REAL' to:")
        print(f"      {modern_ai_path.absolute()}")
        print("   2. Name them: misclassified_001.jpg, misclassified_002.jpg, etc.")
        print("   3. Run this script again")
        return
    
    print(f"\n🖼️ Found {len(modern_ai_images)} modern AI images to add:")
    
    # Copy images to fake training set
    existing_fake_images = list(fake_training_path.glob("*.jpg")) + list(fake_training_path.glob("*.jpeg"))
    next_number = len(existing_fake_images) + 1
    
    copied_count = 0
    
    for img_path in modern_ai_images:
        try:
            # Generate new filename
            new_name = f"modern_ai_{next_number:03d}.jpg"
            destination = fake_training_path / new_name
            
            # Copy file
            shutil.copy2(img_path, destination)
            print(f"   ✅ {img_path.name} → {new_name}")
            
            copied_count += 1
            next_number += 1
            
        except Exception as e:
            print(f"   ❌ Failed to copy {img_path.name}: {e}")
    
    print(f"\n📊 Summary:")
    print(f"   ✅ Successfully copied: {copied_count} images")
    print(f"   📁 Total fake images now: {len(list(fake_training_path.glob('*.jpg'))) + len(list(fake_training_path.glob('*.jpeg')))}")
    
    if copied_count > 0:
        print(f"\n🚀 Next Steps:")
        print(f"   1. Run: python src/data_preprocessing.py")
        print(f"   2. Run: python src/retrain_with_modern_ai.py")
        print(f"   3. Update API and test improved model")
        
        print(f"\n💡 Pro Tip:")
        print(f"   Keep adding misclassified images and retraining!")
        print(f"   This is how production AI systems continuously improve.")

def show_current_dataset_stats():
    """Show current dataset statistics"""
    
    print("\n📊 Current Dataset Statistics:")
    print("-" * 40)
    
    paths = {
        "Real images": Path("data/raw/real"),
        "Fake images": Path("data/raw/fake"), 
        "Modern AI images": Path("data/raw/modern_ai")
    }
    
    for label, path in paths.items():
        if path.exists():
            jpg_count = len(list(path.glob("*.jpg"))) + len(list(path.glob("*.jpeg")))
            png_count = len(list(path.glob("*.png")))
            total = jpg_count + png_count
            print(f"   {label}: {total} images")
        else:
            print(f"   {label}: 0 images (directory not found)")

if __name__ == "__main__":
    show_current_dataset_stats()
    add_misclassified_images() 