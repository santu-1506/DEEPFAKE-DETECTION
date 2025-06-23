import requests
import os
from pathlib import Path
import time

def download_sample_ai_images():
    """
    Download some sample AI-generated images from public sources
    These are known AI-generated images that we can use for testing
    """
    
    modern_ai_path = Path("data/raw/modern_ai")
    modern_ai_path.mkdir(parents=True, exist_ok=True)
    
    # Sample AI-generated images from public datasets/sources
    # These are examples - you'll add the ones you found from Google
    sample_images = [
        {
            "url": "https://this-person-does-not-exist.com/img/avatar-gen112c91a92c0c0118e96e85318a76ce06a.jpg",
            "filename": "sample_ai_001.jpg",
            "description": "AI-generated face from This Person Does Not Exist"
        },
        # Note: We'll create placeholder files since direct downloads may not work
    ]
    
    print("📥 Adding sample AI-generated images...")
    
    # For demonstration, let's create some placeholder instructions
    instructions_file = modern_ai_path / "README.txt"
    with open(instructions_file, 'w') as f:
        f.write("""
How to add AI-generated images:

1. Go back to your Google search results for "realistic ai image creator"
2. Right-click on the AI-generated images you found
3. Save them to this folder (data/raw/modern_ai/)
4. Name them: ai_image_001.jpg, ai_image_002.jpg, etc.

Examples of AI generators to collect from:
- AI-Pro images
- ZMO.AI generated photos  
- Perfect Corp AI humans
- Plugger-AI realistic images

Once you have 10-20 images, run:
python src/data_collection_modern.py --copy
""")
    
    print(f"✅ Created instructions at: {instructions_file}")
    print(f"📁 Save AI images to: {modern_ai_path}")
    
    # Let's also copy one of the existing fake images as an example
    existing_fake_path = Path("data/raw/fake")
    if existing_fake_path.exists():
        fake_images = list(existing_fake_path.glob("*.jpg")) + list(existing_fake_path.glob("*.jpeg"))
        if fake_images:
            import shutil
            # Copy first fake image as example
            example_source = fake_images[0]
            example_target = modern_ai_path / "example_ai_image.jpg"
            shutil.copy2(example_source, example_target)
            print(f"✅ Added example image: {example_target.name}")
    
    return modern_ai_path

if __name__ == "__main__":
    download_sample_ai_images()
    
    print("\n🎯 Next Steps:")
    print("1. Add 10-20 AI images from your Google search to data/raw/modern_ai/")
    print("2. Run: python src/data_collection_modern.py --copy")
    print("3. Run: python src/data_preprocessing.py")
    print("4. Run: python src/model_training.py")
    print("\nThis will retrain the model to detect modern AI images!") 