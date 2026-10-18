"""
Dataset Downloader & Extractor for Polyp Segmentation.

Target environment: Google Colab & Local Machine.
Automatically checks for local dataset.zip or downloads from HuggingFace:
https://huggingface.co/datasets/doantrongthai/Polyp_Segmentation
"""

import os
import zipfile
import shutil
from typing import Optional

DATASET_REPO = 'doantrongthai/Polyp_Segmentation'
DATASET_FILE = 'dataset.zip'
DEFAULT_DATA_ROOT = './data'

LOCAL_CANDIDATE_ZIPS = [
    './dataset.zip',
    '../dataset.zip',
    '../../dataset.zip',
    'D:/Paper/Polyp_Segmentation_Research/dataset.zip',
    '/content/dataset.zip',
    '/content/drive/MyDrive/dataset.zip'
]


def setup_dataset(data_root: str = DEFAULT_DATA_ROOT) -> str:
    """
    Ensure the polyp segmentation dataset is extracted and ready for training & testing.
    
    Expected structure after extraction:
        data_root/
            TrainDataset/
                images/ (1450 images)
                masks/  (1450 masks)
            TestDataset/
                CVC-300/
                CVC-ClinicDB/
                CVC-ColonDB/
                ETIS-LaribPolypDB/
                Kvasir/
    """
    os.makedirs(data_root, exist_ok=True)
    train_images = os.path.join(data_root, 'TrainDataset', 'images')
    if os.path.exists(train_images) and len(os.listdir(train_images)) >= 1000:
        print(f"[Dataset] Verified existing dataset at '{data_root}'. Ready.")
        return data_root

    # Step 1: Check local candidate ZIPs
    found_zip: Optional[str] = None
    for cand in LOCAL_CANDIDATE_ZIPS:
        if os.path.exists(cand) and os.path.getsize(cand) > 100000000:  # >100MB
            found_zip = os.path.abspath(cand)
            print(f"[Dataset] Found local dataset archive at: {found_zip}")
            break

    # Step 2: Download from HuggingFace if no local ZIP
    if not found_zip:
        print(f"[Dataset] Local archive not found. Downloading '{DATASET_FILE}' from HuggingFace repo '{DATASET_REPO}'...")
        try:
            from huggingface_hub import hf_hub_download
            found_zip = hf_hub_download(
                repo_id=DATASET_REPO,
                filename=DATASET_FILE,
                repo_type='dataset'
            )
            print(f"[Dataset] Download completed to cache: {found_zip}")
        except Exception as e:
            print(f"[Dataset] huggingface_hub failed ({e}). Attempting direct download...")
            import requests
            url = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main/{DATASET_FILE}"
            target_zip = os.path.join(data_root, DATASET_FILE)
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(target_zip, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024 * 8):
                        if chunk:
                            f.write(chunk)
            found_zip = target_zip
            print(f"[Dataset] Direct download completed: {found_zip}")

    # Step 3: Extract archive
    print(f"[Dataset] Extracting archive '{found_zip}' to '{data_root}'...")
    with zipfile.ZipFile(found_zip, 'r') as zf:
        zf.extractall(data_root)

    # Step 4: Flatten nested 'dataset' directory if present
    # Some archives contain 'dataset/TrainDataset' instead of 'TrainDataset'
    nested_dir = os.path.join(data_root, 'dataset')
    if os.path.exists(nested_dir) and os.path.isdir(nested_dir):
        for item in os.listdir(nested_dir):
            src = os.path.join(nested_dir, item)
            dst = os.path.join(data_root, item)
            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            shutil.move(src, dst)
        try:
            os.rmdir(nested_dir)
        except Exception:
            pass

    print(f"[Dataset] Extraction and setup finished successfully at '{data_root}'.")
    return data_root


if __name__ == '__main__':
    setup_dataset()
