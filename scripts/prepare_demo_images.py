"""
Script to copy a sample of house images for demo purposes.
This allows including a subset of images in the repository for Streamlit Cloud.
"""

import shutil
from pathlib import Path
import json

def prepare_demo_images(
    num_houses: int = 100,
    source_dir: str = "data/Houses-dataset/Houses Dataset",
    demo_dir: str = "data/Houses-dataset/demo_images",
    associations_path: str = "data/Houses-dataset/associations.json"
):
    """
    Copy images for the first N houses to a demo_images directory.
    
    Args:
        num_houses: Number of houses to include (default: 100)
        source_dir: Source directory with all images
        demo_dir: Destination directory for demo images
        associations_path: Path to associations.json
    """
    source = Path(source_dir)
    demo = Path(demo_dir)
    demo.mkdir(parents=True, exist_ok=True)
    
    # Load associations to get house IDs
    with open(associations_path, 'r') as f:
        associations = json.load(f)
    
    # Get first N houses
    houses_to_include = associations[:num_houses]
    house_ids = [house['house_id'] for house in houses_to_include]
    
    image_types = ['frontal', 'bedroom', 'bathroom', 'kitchen']
    copied_count = 0
    
    print(f"Copying images for {num_houses} houses...")
    
    for house_id in house_ids:
        for img_type in image_types:
            filename = f"{house_id}_{img_type}.jpg"
            source_file = source / filename
            
            if source_file.exists():
                dest_file = demo / filename
                shutil.copy2(source_file, dest_file)
                copied_count += 1
    
    print(f"Copied {copied_count} images to {demo_dir}")
    print(f"\nNext steps:")
    print(f"1. Update .gitignore to allow demo_images/")
    print(f"2. Update display_house_images() to check demo_images/ first")
    print(f"3. Commit and push")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare demo images for Streamlit Cloud")
    parser.add_argument(
        "--num-houses",
        type=int,
        default=100,
        help="Number of houses to include (default: 100)"
    )
    args = parser.parse_args()
    
    prepare_demo_images(num_houses=args.num_houses)

