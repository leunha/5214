import os
import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from tqdm import tqdm
import json

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

def rename_files(directory):
    files = sorted(glob.glob(os.path.join(directory, "*")))
    table = {}
    for idx, file_path in enumerate(tqdm(files)):
        new_name = os.path.join(directory, f"{idx:04d}.npy")
        table[new_name] = file_path

        os.rename(file_path, new_name)

    with open(os.path.join(os.path.dirname(directory), f"{os.path.basename(directory)}_table.json"), "w") as f:
        json.dump(table, f, indent=4)

if __name__ == "__main__":
    process_files("./ixi_dataset/IXI-T1", "./processed_dataset/IXI-T1")
    process_files("./ixi_dataset/IXI-T2", "./processed_dataset/IXI-T2")
    rename_files("./processed_dataset/IXI-T1")
    rename_files("./processed_dataset/IXI-T2")
