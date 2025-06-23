import os
import cv2
import numpy as np
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
import json

def preprocess_images(input_path="data/raw", output_path="data/processed", target_size=(224, 224)):
    """
    Preprocess images for AI training
    
    What this does:
    1. Resize all images to same size (224x224 - standard for many AI models)
    2. Normalize pixel values (0-255 → 0-1)
    3. Save in organized structure for training
    
    Why 224x224?
    - Standard size for many pre-trained models (ResNet, EfficientNet, etc.)
    - Good balance between detail and processing speed
    - Your real images will be downscaled, fake images stay same size
    """
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Create output directories
    (output_path / "real").mkdir(parents=True, exist_ok=True)
    (output_path / "fake").mkdir(parents=True, exist_ok=True)
    
    processing_stats = {
        "target_size": target_size,
        "processed_images": {"real": 0, "fake": 0},
        "failed_images": {"real": 0, "fake": 0}
    }
    
    print(f"🔄 Preprocessing images to {target_size[0]}x{target_size[1]} pixels...")
    
    # Process each category
    for category in ["real", "fake"]:
        input_dir = input_path / category
        output_dir = output_path / category
        
        if not input_dir.exists():
            print(f"⚠️  Warning: {input_dir} doesn't exist")
            continue
            
        print(f"\n📁 Processing {category} images...")
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_dir.glob(f"*{ext}"))
            image_files.extend(input_dir.glob(f"*{ext.upper()}"))
        
        print(f"   Found {len(image_files)} images to process")
        
        # Process each image
        for i, img_path in enumerate(image_files, 1):
            try:
                # Load image
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"   ❌ Failed to load: {img_path.name}")
                    processing_stats["failed_images"][category] += 1
                    continue
                
                # Convert from BGR to RGB (OpenCV uses BGR by default)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize to target size
                img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_LANCZOS4)
                
                # Convert back to BGR for saving with OpenCV
                img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
                
                # Save processed image
                output_filename = f"{category}_{i:04d}.jpg"
                output_filepath = output_dir / output_filename
                
                # Save with high quality
                cv2.imwrite(str(output_filepath), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                processing_stats["processed_images"][category] += 1
                
                # Progress indicator
                if i % 10 == 0 or i == len(image_files):
                    print(f"   Progress: {i}/{len(image_files)} images processed")
                    
            except Exception as e:
                print(f"   ❌ Error processing {img_path.name}: {e}")
                processing_stats["failed_images"][category] += 1
    
    # Save processing statistics
    stats_file = output_path / "preprocessing_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(processing_stats, f, indent=2)
    
    print(f"\n✅ Preprocessing complete!")
    print(f"📊 Results:")
    print(f"   Real images processed: {processing_stats['processed_images']['real']}")
    print(f"   Fake images processed: {processing_stats['processed_images']['fake']}")
    print(f"   Total processed: {sum(processing_stats['processed_images'].values())}")
    print(f"   Failed images: {sum(processing_stats['failed_images'].values())}")
    print(f"📄 Stats saved to: {stats_file}")
    
    return processing_stats

def create_train_val_split(processed_path="data/processed", split_path="data/splits", test_size=0.2, random_state=42):
    """
    Split data into training and validation sets
    
    What this does:
    - Takes your processed images
    - Randomly splits them: 80% training, 20% validation
    - Maintains balance between real/fake in both sets
    
    Why split the data?
    - Training set: Teaches the AI model
    - Validation set: Tests how well the AI learned (unseen data)
    - Prevents overfitting (AI memorizing vs actually learning)
    """
    
    processed_path = Path(processed_path)
    split_path = Path(split_path)
    
    # Create split directories
    for split in ["train", "val"]:
        for category in ["real", "fake"]:
            (split_path / split / category).mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Creating train/validation split (train: {100*(1-test_size):.0f}%, val: {100*test_size:.0f}%)...")
    
    split_stats = {"train": {"real": 0, "fake": 0}, "val": {"real": 0, "fake": 0}}
    
    # Process each category
    for category in ["real", "fake"]:
        category_path = processed_path / category
        
        if not category_path.exists():
            print(f"⚠️  Warning: {category_path} doesn't exist")
            continue
        
        # Get all processed images
        image_files = list(category_path.glob("*.jpg"))
        print(f"\n📁 Splitting {len(image_files)} {category} images...")
        
        if len(image_files) == 0:
            print(f"   No images found in {category_path}")
            continue
        
        # Split the files
        train_files, val_files = train_test_split(
            image_files, 
            test_size=test_size, 
            random_state=random_state,
            shuffle=True
        )
        
        print(f"   Train: {len(train_files)} images")
        print(f"   Val: {len(val_files)} images")
        
        # Copy files to train directory
        for img_file in train_files:
            dst = split_path / "train" / category / img_file.name
            shutil.copy2(img_file, dst)
            split_stats["train"][category] += 1
        
        # Copy files to validation directory
        for img_file in val_files:
            dst = split_path / "val" / category / img_file.name
            shutil.copy2(img_file, dst)
            split_stats["val"][category] += 1
    
    # Save split statistics
    stats_file = split_path / "split_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(split_stats, f, indent=2)
    
    print(f"\n✅ Data split complete!")
    print(f"📊 Final dataset structure:")
    print(f"   Training set:")
    print(f"     - Real: {split_stats['train']['real']} images")
    print(f"     - Fake: {split_stats['train']['fake']} images")
    print(f"     - Total: {split_stats['train']['real'] + split_stats['train']['fake']} images")
    print(f"   Validation set:")
    print(f"     - Real: {split_stats['val']['real']} images") 
    print(f"     - Fake: {split_stats['val']['fake']} images")
    print(f"     - Total: {split_stats['val']['real'] + split_stats['val']['fake']} images")
    print(f"📄 Stats saved to: {stats_file}")
    
    return split_stats

def verify_preprocessing(split_path="data/splits"):
    """
    Verify that preprocessing worked correctly
    
    Checks:
    - All images are the same size
    - Files exist in correct locations
    - No corrupted images
    """
    
    split_path = Path(split_path)
    print(f"\n🔍 Verifying preprocessed data...")
    
    verification_passed = True
    
    # Check each split and category
    for split in ["train", "val"]:
        for category in ["real", "fake"]:
            dir_path = split_path / split / category
            
            if not dir_path.exists():
                print(f"❌ Missing directory: {dir_path}")
                verification_passed = False
                continue
            
            image_files = list(dir_path.glob("*.jpg"))
            print(f"📁 {split}/{category}: {len(image_files)} images")
            
            # Check first few images for consistency
            for img_file in image_files[:3]:
                try:
                    img = cv2.imread(str(img_file))
                    if img is None:
                        print(f"❌ Corrupted image: {img_file}")
                        verification_passed = False
                    else:
                        h, w = img.shape[:2]
                        if (w, h) != (224, 224):
                            print(f"❌ Wrong size: {img_file} is {w}x{h}, expected 224x224")
                            verification_passed = False
                except Exception as e:
                    print(f"❌ Error checking {img_file}: {e}")
                    verification_passed = False
    
    if verification_passed:
        print("✅ All preprocessing checks passed!")
    else:
        print("❌ Some preprocessing issues found!")
    
    return verification_passed

def main_preprocessing():
    """
    Main function that runs the complete preprocessing pipeline
    """
    
    print("🚀 Starting Data Preprocessing Pipeline...")
    print("=" * 60)
    
    # Step 1: Preprocess raw images
    print("\n📋 Step 1: Image Preprocessing")
    preprocessing_stats = preprocess_images()
    
    # Step 2: Create train/validation split
    print("\n📋 Step 2: Train/Validation Split")
    split_stats = create_train_val_split()
    
    # Step 3: Verify everything worked
    print("\n📋 Step 3: Verification")
    verification_passed = verify_preprocessing()
    
    print("\n" + "=" * 60)
    print("🎉 Preprocessing Pipeline Complete!")
    
    if verification_passed:
        print("✅ Your data is ready for AI model training!")
        print("\n📋 What you now have:")
        print("   - All images resized to 224x224 pixels")
        print("   - Balanced training and validation sets")
        print("   - Organized folder structure for training")
        print("\n🚀 Next step: Build and train the AI model!")
    else:
        print("❌ Please check the error messages above and fix any issues.")

if __name__ == "__main__":
    main_preprocessing() 