import os
import requests
from tqdm import tqdm
import pandas as pd
import tarfile
import shutil

def download_file(url, destination):
    """
    Download a file from the provided URL to the destination path
    with a progress bar.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for HTTP errors
    
    # Get file size for progress bar
    file_size = int(response.headers.get('content-length', 0))
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Download with progress bar
    with open(destination, 'wb') as f, tqdm(
            desc=os.path.basename(destination),
            total=file_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                progress_bar.update(len(chunk))
    
    return destination

def extract_tar(tar_path, extract_dir):
    """
    Extract a TAR file to the specified directory.
    """
    try:
        with tarfile.open(tar_path) as tar:
            # Create extraction directory if it doesn't exist
            os.makedirs(extract_dir, exist_ok=True)
            
            # Extract with progress bar
            members = tar.getmembers()
            with tqdm(total=len(members), desc=f"Extracting {os.path.basename(tar_path)}") as progress_bar:
                for member in members:
                    tar.extract(member, path=extract_dir)
                    progress_bar.update(1)
        
        print(f"Successfully extracted {tar_path} to {extract_dir}")
        return True
    except Exception as e:
        print(f"Error extracting {tar_path}: {str(e)}")
        return False

def main():
    if os.path.exists('./ixi_dataset'):
        print("IXI dataset already exists, skipping download.")
        return
    
    # Create main output directory
    os.makedirs('./ixi_dataset', exist_ok=True)
    
    # Define dataset components with actual URLs
    dataset_components = {
        "T1": "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar",
        "T2": "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T2.tar",
        # "PD": "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-PD.tar",
        # "MRA": "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-MRA.tar",
        # "DTI": "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-DTI.tar",
        # "DTI_bvecs": "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/bvecs.txt",
        # "DTI_bvals": "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/bvals.txt",
        # "Demographics": "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI.xls"
    }
    
    print("Downloading IXI dataset...")
    
    # Download each component
    for component_name, component_url in dataset_components.items():
        file_name = os.path.basename(component_url)
        destination_path = os.path.join('./ixi_dataset', file_name)
        
        # Check if file already exists
        if os.path.exists(destination_path):
            print(f"{component_name} data already exists at {destination_path}, skipping download.")
            continue
        
        print(f"Downloading {component_name} data...")
        try:
            download_file(component_url, destination_path)
            print(f"Successfully downloaded {component_name} data to {destination_path}")
        except Exception as e:
            print(f"Error downloading {component_name} data: {str(e)}")
    
    print("Download completed!")
    
    # Extract TAR files if requested
    print("\nExtracting downloaded TAR files...")
    for component_name, component_url in dataset_components.items():
        file_name = os.path.basename(component_url)
        file_path = os.path.join('./ixi_dataset', file_name)
        
        # Skip if file doesn't exist or isn't a TAR file
        if not os.path.exists(file_path) or not file_name.endswith('.tar'):
            continue
        
        # Create extraction directory
        extract_dir = os.path.join('./ixi_dataset', file_name.replace('.tar', ''))
        
        # Check if already extracted
        if os.path.exists(extract_dir) and os.listdir(extract_dir):
            print(f"{file_name} appears to be already extracted, skipping.")
            continue
        
        print(f"Extracting {file_name}...")
        success = extract_tar(file_path, extract_dir)
    
    print("Extraction completed!")

if __name__ == "__main__":
    main()
