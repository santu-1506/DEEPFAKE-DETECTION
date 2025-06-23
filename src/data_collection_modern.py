import os
import requests
import shutil
from pathlib import Path
import time

def create_modern_ai_dataset_structure():
    """Create directories for modern AI-generated images"""
    
    # Create new directories for modern AI images
    modern_ai_path = Path("data/raw/modern_ai")
    modern_ai_path.mkdir(parents=True, exist_ok=True)
    
    print("📁 Created directory structure for modern AI images")
    print(f"   📂 {modern_ai_path}")
    
    return modern_ai_path

def download_sample_ai_images():
    """
    Download some sample AI-generated images from public sources
    Note: In practice, you'd collect these from the AI generators you found
    """
    
    modern_ai_path = create_modern_ai_dataset_structure()
    
    # Sample URLs of known AI-generated images (these are examples)
    # In practice, you'd save the images you found from Google search
    sample_ai_urls = [
        # These would be replaced with actual AI-generated image URLs
        # For now, we'll create placeholders
    ]
    
    print("💡 To add the images you found:")
    print("   1. Save the AI images from Google search to:", modern_ai_path)
    print("   2. Name them: modern_ai_001.jpg, modern_ai_002.jpg, etc.")
    print("   3. Then run the retraining script")
    
    return modern_ai_path

def copy_images_to_fake_dataset(source_dir, target_dir="data/raw/fake"):
    """
    Copy modern AI images to the fake dataset for retraining
    """
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    if not source_path.exists():
        print(f"❌ Source directory doesn't exist: {source_path}")
        return
    
    # Get existing fake images count to continue numbering
    existing_fake = list(target_path.glob("*.jpg")) + list(target_path.glob("*.jpeg"))
    next_number = len(existing_fake) + 1
    
    copied_count = 0
    
    # Copy all images from modern_ai directory
    for img_file in source_path.glob("*"):
        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp', '.avif']:
            # Generate new filename
            new_filename = f"modern_ai_{next_number:03d}{img_file.suffix}"
            target_file = target_path / new_filename
            
            # Copy the file
            shutil.copy2(img_file, target_file)
            print(f"✅ Copied: {img_file.name} -> {new_filename}")
            
            next_number += 1
            copied_count += 1
    
    print(f"📊 Total modern AI images added: {copied_count}")
    return copied_count

def main():
    """Main function to set up modern AI image collection"""
    
    print("🚀 Setting up Modern AI Image Collection")
    print("=" * 50)
    
    # Step 1: Create directory structure
    modern_ai_path = create_modern_ai_dataset_structure()
    
    # Step 2: Instructions for manual collection
    print("\n📋 Next Steps:")
    print("1. Save the AI images you found from Google to:", modern_ai_path)
    print("2. Supported formats: JPG, PNG, WebP, AVIF")
    print("3. Name them descriptively (e.g., ai_pro_001.jpg, zmo_ai_002.jpg)")
    print("4. Run this script again with --copy to add them to training data")
    
    print(f"\n💡 Example command to save an image:")
    print(f"   Right-click image -> Save As -> {modern_ai_path}/ai_image_001.jpg")
    
    return modern_ai_path

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--copy":
        # Copy images from modern_ai to fake dataset
        modern_ai_path = Path("data/raw/modern_ai")
        if modern_ai_path.exists():
            copied = copy_images_to_fake_dataset(modern_ai_path)
            if copied > 0:
                print(f"\n🎯 Ready to retrain! Added {copied} modern AI images.")
                print("   Run: python src/data_preprocessing.py")
                print("   Then: python src/model_training.py")
        else:
            print("❌ No modern AI images found. Please add some first.")
    else:
        main() 