"""
Data Download Script for NSL-KDD Dataset
Downloads the dataset from GitHub repository
"""

import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RAW_DATA_DIR, DATASET_CONFIG


def download_file(url: str, filepath: Path, desc: str = "Downloading") -> bool:
    """
    Download a file from URL with progress bar.
    
    Args:
        url: URL to download from
        filepath: Path to save the file
        desc: Description for progress bar
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"✓ Downloaded: {filepath.name}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Error downloading {url}: {e}")
        return False


def download_nsl_kdd_dataset() -> bool:
    """
    Download the NSL-KDD dataset files.
    
    Returns:
        bool: True if all files downloaded successfully
    """
    print("\n" + "="*60)
    print("  NSL-KDD Dataset Downloader")
    print("="*60 + "\n")
    
    # Ensure directory exists
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    files_to_download = [
        (DATASET_CONFIG['train_url'], DATASET_CONFIG['train_file'], "Training Set"),
        (DATASET_CONFIG['test_url'], DATASET_CONFIG['test_file'], "Test Set")
    ]
    
    success = True
    for url, filename, desc in files_to_download:
        filepath = RAW_DATA_DIR / filename
        
        if filepath.exists():
            print(f"ℹ Already exists: {filename}")
            continue
            
        if not download_file(url, filepath, desc):
            success = False
    
    if success:
        print("\n✓ All dataset files are ready!")
        print(f"  Location: {RAW_DATA_DIR}")
    else:
        print("\n✗ Some files failed to download.")
        
    return success


def verify_dataset() -> dict:
    """
    Verify the downloaded dataset files.
    
    Returns:
        dict: Information about the dataset files
    """
    info = {}
    
    for key in ['train_file', 'test_file']:
        filepath = RAW_DATA_DIR / DATASET_CONFIG[key]
        
        if filepath.exists():
            # Count lines
            with open(filepath, 'r') as f:
                line_count = sum(1 for _ in f)
            
            info[key] = {
                'exists': True,
                'path': str(filepath),
                'size_mb': filepath.stat().st_size / (1024 * 1024),
                'records': line_count
            }
        else:
            info[key] = {'exists': False}
    
    return info


if __name__ == "__main__":
    # Download dataset
    download_nsl_kdd_dataset()
    
    # Verify
    print("\n" + "-"*60)
    print("  Dataset Verification")
    print("-"*60)
    
    info = verify_dataset()
    for name, details in info.items():
        if details['exists']:
            print(f"\n  {name}:")
            print(f"    Records: {details['records']:,}")
            print(f"    Size: {details['size_mb']:.2f} MB")
        else:
            print(f"\n  {name}: NOT FOUND")
