import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import argparse
from tqdm import tqdm
import os
import glob
import nibabel as nib

def display_paired_images(t1_array, t2_array):
    # Create figure and subplots with space for the sliders
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    plt.subplots_adjust(bottom=0.2)  # Make room for sliders
    
    # Display initial frames (frame 0)
    im1 = ax1.imshow(t1_array[..., 0], cmap='gray')
    plt.colorbar(im1, ax=ax1)
    ax1.axis('off')
    ax1.set_title('T1')
    
    im2 = ax2.imshow(t2_array[..., 0], cmap='gray')
    plt.colorbar(im2, ax=ax2)
    ax2.axis('off')
    ax2.set_title('T2')
    
    # Add sliders
    slider_ax1 = plt.axes([0.1, 0.05, 0.3, 0.03])  # [left, bottom, width, height]
    slider_ax2 = plt.axes([0.6, 0.05, 0.3, 0.03])
    
    frame_slider1 = Slider(
        ax=slider_ax1,
        label='T1 Frame',
        valmin=0,
        valmax=t1_array.shape[-1] - 1,
        valinit=0,
        valstep=1
    )
    
    frame_slider2 = Slider(
        ax=slider_ax2,
        label='T2 Frame',
        valmin=0,
        valmax=t2_array.shape[-1] - 1,
        valinit=0,
        valstep=1
    )
    
    
    # Update functions for sliders
    def update1(val):
        frame = int(frame_slider1.val)
        im1.set_array(t1_array[..., frame])
        fig.canvas.draw_idle()
    
    def update2(val):
        frame = int(frame_slider2.val)
        im2.set_array(t2_array[..., frame])
        fig.canvas.draw_idle()
    
    frame_slider1.on_changed(update1)
    frame_slider2.on_changed(update2)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Display paired T1/T2 images')
    parser.add_argument('-f', '--filename', type=str, help='Image filename (without path and extension)')
    parser.add_argument('--processed', action='store_true', help='Use processed NumPy files instead of NIFTI')
    args = parser.parse_args()

    if args.processed:
        # Use processed NumPy files
        print("Using processed NumPy files")
        
        # Look for matching files in the processed directories
        # First check if it's a numeric index
        if args.filename.isdigit():
            file_index = args.filename.zfill(4)  # Pad with zeros to 4 digits
            t1_path = os.path.join('processed_dataset', 'IXI-T1', f"{file_index}.npy")
            t2_path = os.path.join('processed_dataset', 'IXI-T2', f"{file_index}.npy")
        else:
            # Show available files and exit
            print("For processed files, please provide a numeric index (0-2):")
            t1_files = sorted(glob.glob("./processed_dataset/IXI-T1/*.npy"))[:5]
            t2_files = sorted(glob.glob("./processed_dataset/IXI-T2/*.npy"))[:5]
            print("Available T1 files:")
            for i, f in enumerate(t1_files):
                print(f"  {i}: {os.path.basename(f)}")
            print("Available T2 files:")
            for i, f in enumerate(t2_files):
                print(f"  {i}: {os.path.basename(f)}")
            exit(1)
        
        # Load NumPy arrays
        print(f"Loading T1 file: {t1_path}")
        print(f"Loading T2 file: {t2_path}")
        
        t1_array = np.load(t1_path)
        t2_array = np.load(t2_path)
        
        # Transpose to have the slice dimension last if needed
        if t1_array.shape[2] > t1_array.shape[0]:
            t1_array = np.transpose(t1_array, (1, 0, 2))
        if t2_array.shape[2] > t2_array.shape[0]:
            t2_array = np.transpose(t2_array, (1, 0, 2))
    else:
        # Use original NIFTI files
        print("Using original NIFTI files")
        
        # Find matching files in T1 and T2 directories
        t1_files = glob.glob(f"./IXI-T1/*{args.filename}*T1.nii.gz")
        t2_files = glob.glob(f"./IXI-T2 2/*{args.filename}*T2.nii.gz")
        
        if not t1_files or not t2_files:
            print(f"No matching files found for {args.filename}")
            # List some available files as examples
            t1_sample = glob.glob("./IXI-T1/*.nii.gz")[:5]
            print("Some available T1 files:")
            for f in t1_sample:
                print(f"  {os.path.basename(f)}")
            exit(1)
        
        # Use the first matching file
        t1_path = t1_files[0]
        t2_path = t2_files[0]
        
        print(f"Loading T1 file: {t1_path}")
        print(f"Loading T2 file: {t2_path}")
        
        # Load NIFTI files directly
        t1_nifti = nib.load(t1_path)
        t2_nifti = nib.load(t2_path)
        
        # Get data arrays
        t1_array = t1_nifti.get_fdata()
        t2_array = t2_nifti.get_fdata()
        
        # Transpose to have the slice dimension last
        t1_array = np.transpose(t1_array, (1, 0, 2))
        t2_array = np.transpose(t2_array, (1, 0, 2))
    
    print("T1 array shape:", t1_array.shape)
    print("T2 array shape:", t2_array.shape)
    print("T1 value range:", t1_array.min(), t1_array.max())
    print("T2 value range:", t2_array.min(), t2_array.max())
    
    display_paired_images(t1_array, t2_array)
