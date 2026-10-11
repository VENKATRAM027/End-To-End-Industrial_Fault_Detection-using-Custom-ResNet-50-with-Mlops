import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

from PIL import Image, UnidentifiedImageError
from industrial_fault_detection.config import PROCESSED_DATA_DIR, CATEGORY

def validate_dataset():
    category_dir = PROCESSED_DATA_DIR / CATEGORY
    if not category_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {category_dir}")

    for split in ["train", "val"]:
        for cls in ["good", "anomaly"]:
            folder_path = category_dir / split / cls
            if not folder_path.exists():
                raise FileNotFoundError(f"Expected folder missing: {folder_path}")
            
            for img_path in list(folder_path.glob("*.png")):
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                except (UnidentifiedImageError, OSError):
                    print(f"Removing corrupted image: {img_path}")
                    img_path.unlink()
    print("Dataset validation passed.")

if __name__ == "__main__":
    validate_dataset()