import os
import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from tqdm import tqdm

def process_files(raw_directory, processed_directory):
    """
    Process all files in the specified directory.
    
    Args:
        raw_directory (str): Path to the raw directory
        processed_directory (str): Path to the processed directory
    """
    files = sorted(glob.glob(os.path.join(raw_directory, "*")))
    os.makedirs(processed_directory, exist_ok=True)

    # Process each file
    for file_path in tqdm(files):
        data = nib.load(file_path)
        np.save(os.path.join(processed_directory, os.path.basename(file_path).replace(".nii.gz", ".npy")), data.get_fdata())

if __name__ == "__main__":
    process_files("./ixi_dataset/IXI-T1", "./processed_dataset/IXI-T1")
    process_files("./ixi_dataset/IXI-T2", "./processed_dataset/IXI-T2")
