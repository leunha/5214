import os
import glob
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

def process_ixi_dataset(t1_dir, t2_dir, output_dir, num_subjects=None, registration=True):
    """
    Process the IXI dataset by aligning and resampling T1 and T2 volumes.
    
    Args:
        t1_dir: Directory containing T1 NIfTI files
        t2_dir: Directory containing T2 NIfTI files
        output_dir: Directory to save processed numpy files
        num_subjects: Number of subjects to process (None for all)
        registration: Whether to perform registration between T1 and T2
    """
    # Find all NIfTI files (.nii or .nii.gz)
    t1_files = sorted(glob.glob(os.path.join(t1_dir, '*.nii*')))
    t2_files = sorted(glob.glob(os.path.join(t2_dir, '*.nii*')))
    
    # Extract subject IDs from filenames
    t1_subjects = ['-'.join(os.path.basename(f).split('-')[:2]) for f in t1_files]
    t2_subjects = ['-'.join(os.path.basename(f).split('-')[:2]) for f in t2_files]
    
    # Find common subjects
    common_subjects = set(t1_subjects).intersection(set(t2_subjects))
    print(f"Found {len(common_subjects)} subjects with both T1 and T2 scans")
    
    # Limit number of subjects if specified
    if num_subjects != 0:
        common_subjects = list(common_subjects)[:num_subjects]
        print(f"Processing {len(common_subjects)} subjects")
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'IXI-T1'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'IXI-T2'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    
    # Process each subject
    for subject_id in tqdm(common_subjects, desc="Processing subjects"):
        # Find matching files
        t1_file = [f for f, s in zip(t1_files, t1_subjects) if s == subject_id][0]
        t2_file = [f for f, s in zip(t2_files, t2_subjects) if s == subject_id][0]
        
        try:
            # Load images using SimpleITK
            t1_img = sitk.ReadImage(t1_file)
            t2_img = sitk.ReadImage(t2_file)
            
            # Resample to isotropic resolution (1mm)
            original_spacing = t1_img.GetSpacing()
            new_spacing = [1.0, 1.0, 1.0]
            
            resample = sitk.ResampleImageFilter()
            resample.SetInterpolator(sitk.sitkLinear)
            resample.SetOutputSpacing(new_spacing)
            resample.SetOutputDirection(t1_img.GetDirection())
            resample.SetOutputOrigin(t1_img.GetOrigin())
            
            new_size = [int(round(osz*ospc/nspc)) for osz,ospc,nspc in zip(t1_img.GetSize(), original_spacing, new_spacing)]
            resample.SetSize(new_size)
            
            t1_resampled = resample.Execute(t1_img)
            t2_resampled = resample.Execute(t2_img)
            
            if registration:
                # Register T2 to T1
                elastixImageFilter = sitk.ElastixImageFilter()
                elastixImageFilter.SetFixedImage(t1_resampled)
                elastixImageFilter.SetMovingImage(t2_resampled)
                
                parameterMap = sitk.GetDefaultParameterMap('affine')
                elastixImageFilter.SetParameterMap(parameterMap)
                
                t2_registered = elastixImageFilter.Execute()
            else:
                t2_registered = t2_resampled
            
            # Convert to numpy arrays
            t1_np = sitk.GetArrayFromImage(t1_resampled)
            t2_np = sitk.GetArrayFromImage(t2_registered)
            
            # Transpose T2 volumes for better visualization (instead of T1)
            t2_np = np.transpose(t2_np, (1, 0, 2))
            
            # Normalize volumes to [0, 1]
            t1_np = (t1_np - t1_np.min()) / (t1_np.max() - t1_np.min() + 1e-8)
            t2_np = (t2_np - t2_np.min()) / (t2_np.max() - t2_np.min() + 1e-8)
            
            # Visualize central slices to verify alignment
            if subject_id == list(common_subjects)[0] or subject_id == list(common_subjects)[-1]:
                plot_alignment_check(t1_np, t2_np, subject_id, output_dir)
            
            # Save as numpy files
            np.save(os.path.join(output_dir, 'IXI-T1', f"{subject_id}.npy"), t1_np)
            np.save(os.path.join(output_dir, 'IXI-T2', f"{subject_id}.npy"), t2_np)
            
        except Exception as e:
            print(f"Error processing {subject_id}: {str(e)}")

def plot_alignment_check(t1_volume, t2_volume, subject_id, output_dir):
    """
    Create a visualization showing alignment between T1 and T2 volumes
    with multiple slices and different views to verify correspondence.
    """
    # Get volume shape
    depth, height, width = t1_volume.shape
    
    # Transpose T2 volume for visualization comparison
    t2_volume_transposed = np.transpose(t2_volume, (1, 0, 2))
    
    # Create figure with multiple views (axial, coronal, sagittal)
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    
    # Set titles for rows and columns
    axes[0, 0].set_title("T1 Image")
    axes[0, 1].set_title("T2 Image")
    axes[0, 2].set_title("T2 Transposed")
    axes[0, 3].set_title("T1-T2 Overlay")
    
    views = ["Axial", "Coronal", "Sagittal"]
    
    # Plot each view
    for i, view in enumerate(views):
        axes[i, 0].set_ylabel(view)
        
        # Select slice indices for each view
        if view == "Axial":
            slice_idx = depth // 2
            t1_slice = t1_volume[slice_idx, :, :]
            t2_slice = t2_volume[slice_idx, :, :]
            t2_slice_transposed = t2_volume_transposed[slice_idx, :, :]
        elif view == "Coronal":
            slice_idx = height // 2
            t1_slice = t1_volume[:, slice_idx, :]
            t2_slice = t2_volume[:, slice_idx, :]
            t2_slice_transposed = t2_volume_transposed[:, slice_idx, :]
        else:  # Sagittal
            slice_idx = width // 2
            t1_slice = t1_volume[:, :, slice_idx]
            t2_slice = t2_volume[:, :, slice_idx]
            t2_slice_transposed = t2_volume_transposed[:, :, slice_idx]
        
        # Normalize slices for visualization
        t1_norm = (t1_slice - t1_slice.min()) / (t1_slice.max() - t1_slice.min() + 1e-8)
        t2_norm = (t2_slice - t2_slice.min()) / (t2_slice.max() - t2_slice.min() + 1e-8)
        t2_transposed_norm = (t2_slice_transposed - t2_slice_transposed.min()) / (t2_slice_transposed.max() - t2_slice_transposed.min() + 1e-8)
        
        # Display T1
        axes[i, 0].imshow(t1_norm, cmap='gray')
        axes[i, 0].axis('off')
        
        # Display T2
        axes[i, 1].imshow(t2_norm, cmap='gray')
        axes[i, 1].axis('off')
        
        # Display Transposed T2
        axes[i, 2].imshow(t2_transposed_norm, cmap='gray')
        axes[i, 2].axis('off')
        
        # Display overlay
        axes[i, 3].imshow(t1_norm, cmap='gray')
        axes[i, 3].imshow(t2_transposed_norm, cmap='hot', alpha=0.5)
        axes[i, 3].axis('off')
    
    plt.suptitle(f"T1-T2 Alignment Check for Subject {subject_id}", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'visualizations', f"{subject_id}_alignment_check.png"))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process IXI dataset for MRI translation")
    parser.add_argument("--t1_dir", type=str, default="./IXI-T1", help="Directory with T1 NIfTI files")
    parser.add_argument("--t2_dir", type=str, default="./IXI-T2 2", help="Directory with T2 NIfTI files")
    parser.add_argument("--output_dir", type=str, default="./processed_dataset", help="Directory to save processed files")
    parser.add_argument("--num_subjects", type=int, default=5, help="Number of subjects to process (default: 5)")
    parser.add_argument("--no_registration", action="store_false", dest="registration", help="Skip registration step")
    args = parser.parse_args()
    
    process_ixi_dataset(args.t1_dir, args.t2_dir, args.output_dir, args.num_subjects, args.registration)
