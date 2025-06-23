# src/data_collection.py
import os
from pathlib import Path

def setup_data_directories():
    """
    Creates the folder structure for organizing our deepfake detection data
    
    Think of this like organizing a filing cabinet:
    - Raw folder = incoming mail (unsorted)
    - Processed folder = filed documents (ready to use)
    """
    # Define our main data directory
    base_dir = Path("data")
    
    # Create all the folders we need
    directories = [
        "data/raw/real",      # For authentic photos
        "data/raw/fake",      # For deepfake images
        "data/processed/real", # For cleaned authentic photos
        "data/processed/fake"  # For cleaned deepfake images
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Create a simple info file
    info_content = """
# Deepfake Detection Dataset

## Directory Structure:
- data/raw/real/     -> Place authentic human photos here
- data/raw/fake/     -> Place AI-generated deepfake images here
- data/processed/    -> Automatically created clean images go here

## Data Collection Guidelines:
1. Real images: High-quality photos of faces (well-lit, clear)
2. Fake images: AI-generated faces from tools like ThisPersonDoesNotExist
3. Aim for 100-500 images in each category to start
4. Keep images in JPG/PNG format
"""
    
    with open("data/README.md", "w") as f:
        f.write(info_content)
    
    print("\n🎉 Data structure created successfully!")
    print("📖 Check data/README.md for collection guidelines")
    
    return base_dir

if __name__ == "__main__":
    setup_data_directories()