import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

import shutil
import random
import kagglehub
from industrial_fault_detection.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, CATEGORY, RANDOM_SEED

def ingest_data():
    print("Fetching MVTec AD dataset from Kaggle...")
    raw_cache_path = kagglehub.dataset_download("ipythonx/mvtec-ad") 
    
    category_raw_path = Path(raw_cache_path) / CATEGORY
    if not category_raw_path.exists():
        raise FileNotFoundError(f"Category '{CATEGORY}' not found in downloaded dataset.")

    train_good_dir = PROCESSED_DATA_DIR / CATEGORY / "train" / "good" 
    train_anom_dir = PROCESSED_DATA_DIR / CATEGORY / "train" / "anomaly" 
    val_good_dir = PROCESSED_DATA_DIR / CATEGORY / "val" / "good" 
    val_anom_dir = PROCESSED_DATA_DIR / CATEGORY / "val" / "anomaly" 

    print(f"Structuring dataset for {CATEGORY} into {PROCESSED_DATA_DIR}...")
    for d in [train_good_dir, train_anom_dir, val_good_dir, val_anom_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True) #[cite: 1]

    good_images = []
    if (category_raw_path / "train" / "good").exists():
        good_images.extend(list((category_raw_path / "train" / "good").glob("*.png"))) 
    if (category_raw_path / "test" / "good").exists():
        good_images.extend(list((category_raw_path / "test" / "good").glob("*.png"))) 

    anomaly_images = []
    test_dir = category_raw_path / "test" 
    if test_dir.exists():
        for defect_folder in test_dir.iterdir():
            if defect_folder.is_dir() and defect_folder.name != "good":
                anomaly_images.extend(list(defect_folder.glob("*.png")))

    random.seed(RANDOM_SEED)
    random.shuffle(good_images)
    random.shuffle(anomaly_images) 

    split_good = int(len(good_images) * 0.8) 
    split_anom = int(len(anomaly_images) * 0.8) 

    def copy_files(file_list, dest_dir):
        for f in file_list:
            dest_name = f"{f.parent.name}_{f.name}"
            shutil.copy2(f, dest_dir / dest_name) 

    copy_files(good_images[:split_good], train_good_dir) 
    copy_files(good_images[split_good:], val_good_dir) 
    copy_files(anomaly_images[:split_anom], train_anom_dir) 
    copy_files(anomaly_images[split_anom:], val_anom_dir) 

    print("Success! Processed dataset is ready.") 

if __name__ == "__main__":
    ingest_data() 