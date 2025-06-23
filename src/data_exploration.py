import os
import cv2  # For image processing
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For creating visualizations
from pathlib import Path  # For file path handling

def count_dataset_files(data_path="data/raw"):
    """
    Count how many real and fake images we have
    
    Why this matters:
    - We need balanced data (similar amounts of real vs fake)
    - Helps us track collection progress
    - Identifies if we need more of either type
    """
    
    data_path = Path(data_path)
    
    # Count files in each category
    real_files = list((data_path / "real").glob("*"))
    fake_files = list((data_path / "fake").glob("*"))
    
    # Filter for actual image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    real_images = [f for f in real_files if f.suffix.lower() in image_extensions]
    fake_images = [f for f in fake_files if f.suffix.lower() in image_extensions]
    
    print("📊 Dataset Statistics:")
    print(f"   Real images: {len(real_images)}")
    print(f"   Fake images: {len(fake_images)}")
    print(f"   Total images: {len(real_images) + len(fake_images)}")
    
    # Check balance
    if len(real_images) > 0 and len(fake_images) > 0:
        ratio = len(real_images) / len(fake_images)
        if 0.8 <= ratio <= 1.2:
            print("✅ Dataset is well balanced!")
        else:
            print(f"⚠️  Dataset imbalance detected (ratio: {ratio:.2f})")
            if ratio > 1.2:
                print("   Suggestion: Add more fake images")
            else:
                print("   Suggestion: Add more real images")
    
    return real_images, fake_images

def visualize_sample_images(real_images, fake_images, num_samples=4):
    """
    Show sample images from our dataset
    
    Why this helps:
    - Visual inspection of data quality
    - Spot obvious problems (blurry, wrong content, etc.)
    - Understand what our AI will be learning from
    """
    
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 8))
    fig.suptitle('Dataset Sample Images', fontsize=16)
    
    # Show real image samples
    print("\n🔍 Showing sample images...")
    for i in range(min(num_samples, len(real_images))):
        img_path = real_images[i]
        
        # Load and convert image (OpenCV loads in BGR, we need RGB for display)
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axes[0, i].imshow(img_rgb)
        axes[0, i].set_title(f"Real {i+1}\n{img_path.name}", fontsize=10)
        axes[0, i].axis('off')  # Remove axis numbers
        
        # Print image info
        print(f"   Real {i+1}: {img.shape} pixels, File: {img_path.name}")
    
    # Show fake image samples  
    for i in range(min(num_samples, len(fake_images))):
        img_path = fake_images[i]
        
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axes[1, i].imshow(img_rgb)
        axes[1, i].set_title(f"Fake {i+1}\n{img_path.name}", fontsize=10)
        axes[1, i].axis('off')
        
        print(f"   Fake {i+1}: {img.shape} pixels, File: {img_path.name}")
    
    plt.tight_layout()
    
    # Save the visualization
    output_path = "docs/dataset_samples.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Sample visualization saved to: {output_path}")
    
    plt.show()

def analyze_image_properties(real_images, fake_images):
    """
    Analyze technical properties of our images
    
    What we're checking:
    - Image sizes (consistency is important)
    - Color channels (should be 3 for RGB)
    - File sizes (very small files might be low quality)
    """
    
    def get_image_stats(image_list, category_name):
        sizes = []
        file_sizes = []
        
        for img_path in image_list[:20]:  # Check first 20 to save time
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    h, w = img.shape[:2]
                    sizes.append((w, h))
                    file_sizes.append(img_path.stat().st_size / 1024)  # KB
            except Exception as e:
                print(f"   Error reading {img_path}: {e}")
        
        if sizes:
            widths, heights = zip(*sizes)
            print(f"\n📏 {category_name} Image Analysis:")
            print(f"   Average size: {np.mean(widths):.0f} x {np.mean(heights):.0f} pixels")
            print(f"   Size range: {min(widths)}x{min(heights)} to {max(widths)}x{max(heights)}")
            print(f"   Average file size: {np.mean(file_sizes):.1f} KB")
    
    get_image_stats(real_images, "Real")
    get_image_stats(fake_images, "Fake")

def main_exploration():
    """
    Main function that runs all our data exploration
    """
    
    print("🚀 Starting Data Exploration...")
    print("=" * 50)
    
    # Step 1: Count files
    real_images, fake_images = count_dataset_files()
    
    if len(real_images) == 0 and len(fake_images) == 0:
        print("\n❌ No images found!")
        print("📝 Action needed:")
        print("   1. Add real images to data/raw/real/")
        print("   2. Add fake images to data/raw/fake/")
        print("   3. Run this script again")
        return
    
    # Step 2: Visual inspection
    if len(real_images) > 0 or len(fake_images) > 0:
        visualize_sample_images(real_images, fake_images)
    
    # Step 3: Technical analysis
    analyze_image_properties(real_images, fake_images)
    
    print("\n✅ Data exploration complete!")
    print("📋 Next steps:")
    print("   1. Review the sample images")
    print("   2. Add more data if needed")
    print("   3. Run data preprocessing")

if __name__ == "__main__":
    # Make sure docs directory exists for saving plots
    Path("docs").mkdir(exist_ok=True)
    main_exploration()