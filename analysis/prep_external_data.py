import os
import glob
from pathlib import Path
from PIL import Image

def prep_external_data(input_dir, output_dir, target_size=(128, 128)):
    """
    Reads external natural images (like DIV2K/BSD400), converts them to grayscale,
    center-crops them, and resizes to target_size to act as OOD texture injections.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    exts = ("*.png", "*.jpg", "*.jpeg")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(input_dir, "**", ext), recursive=True))
        
    print(f"Found {len(files)} images in {input_dir}. Processing...")
    
    for i, filepath in enumerate(files):
        try:
            img = Image.open(filepath).convert("L")
            
            # Center crop to square
            width, height = img.size
            new_size = min(width, height)
            left = (width - new_size) // 2
            top = (height - new_size) // 2
            right = (width + new_size) // 2
            bottom = (height + new_size) // 2
            
            img = img.crop((left, top, right, bottom))
            
            # Resize
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            
            stem = Path(filepath).stem
            img.save(out_path / f"ext_{stem}.png")
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(files)}")
        except Exception as e:
            print(f"Failed to process {filepath}: {e}")
            
    print("Done! Mix these into your training folder.")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True, help="Path to raw DIV2K/BSD400 images")
    p.add_argument("--output_dir", required=True, help="Path to save processed grayscale patches")
    args = p.parse_args()
    
    prep_external_data(args.input_dir, args.output_dir)
