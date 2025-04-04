import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms
import argparse

class MRISliceDataset(Dataset):
    """
    Dataset for paired 2D MRI slices (T1 and T2), using only middle 30% of slices
    """
    def __init__(self, t1_dir, t2_dir, transform=None, normalize=True, slice_axis=2, target_size=(256, 256), middle_percent=0.3):
        """
        Initialize the dataset.
        
        Args:
            t1_dir: Directory containing T1 volumes
            t2_dir: Directory containing T2 volumes
            transform: Optional transformations to apply
            normalize: Whether to normalize images to [0, 1]
            slice_axis: Axis to slice along (default: 2 for axial slices)
            target_size: Target size for all slices (height, width)
            middle_percent: Percentage of middle slices to use (default: 0.3 for 30%)
        """
        self.t1_files = sorted(glob.glob(os.path.join(t1_dir, "*.npy")))
        self.t2_files = sorted(glob.glob(os.path.join(t2_dir, "*.npy")))
        
        # Extract volume identifiers
        self.t1_identifiers = [os.path.basename(f).split('.')[0] for f in self.t1_files]
        self.t2_identifiers = [os.path.basename(f).split('.')[0] for f in self.t2_files]
        
        # Find common identifiers
        common_identifiers = set(self.t1_identifiers).intersection(set(self.t2_identifiers))
        
        # Filter to keep only pairs
        self.t1_files = [f for f, identifier in zip(self.t1_files, self.t1_identifiers) 
                         if identifier in common_identifiers]
        self.t2_files = [f for f, identifier in zip(self.t2_files, self.t2_identifiers)
                         if identifier in common_identifiers]
        
        # Sort to ensure correspondence
        self.t1_files.sort()
        self.t2_files.sort()
        
        self.transform = transform
        self.normalize = normalize
        self.slice_axis = slice_axis
        self.target_size = target_size
        self.middle_percent = middle_percent
        
        # Create index mapping for efficient slice access
        self.slice_mapping = []
        total_slices = 0
        
        for i in range(len(self.t1_files)):
            t1_volume = np.load(self.t1_files[i])
            t2_volume = np.load(self.t2_files[i])

            # Transpose T1 volume
            # t1_volume = np.transpose(t1_volume, (1, 0, 2))
            
            # Get number of slices
            num_slices = min(t1_volume.shape[self.slice_axis], 
                           t2_volume.shape[self.slice_axis])
            
            # Calculate middle slice range (30% of total)
            middle_slices = int(num_slices * middle_percent)
            start_idx = (num_slices - middle_slices) // 2
            end_idx = start_idx + middle_slices
            
            # Store mapping of each slice to its volume
            for slice_idx in range(start_idx, end_idx):
                self.slice_mapping.append((i, slice_idx))
            total_slices += middle_slices
        
        print(f"Dataset contains {len(self.t1_files)} paired volumes")
        print(f"Using middle {middle_percent*100:.1f}% of slices ({total_slices} total slices)")
        print(f"Average {total_slices/len(self.t1_files):.1f} slices per volume")
    
    def __len__(self):
        return len(self.slice_mapping)
    
    def __getitem__(self, idx):
        # Get volume and slice indices from mapping
        volume_idx, slice_idx = self.slice_mapping[idx]
        
        # Load the T1 and T2 volumes
        t1_volume = np.load(self.t1_files[volume_idx])
        t2_volume = np.load(self.t2_files[volume_idx])

        # Transpose T1 volume
        t1_volume = np.transpose(t1_volume, (1, 0, 2))
        
        
        # Extract the slices based on the axis
        if self.slice_axis == 0:
            t1_slice = t1_volume[slice_idx, :, :]
            t2_slice = t2_volume[slice_idx, :, :]
        elif self.slice_axis == 1:
            t1_slice = t1_volume[:, slice_idx, :]
            t2_slice = t2_volume[:, slice_idx, :]
        else:  # self.slice_axis == 2
            t1_slice = t1_volume[:, :, slice_idx]
            t2_slice = t2_volume[:, :, slice_idx]
        
        # Ensure slices are 2D
        t1_slice = np.asarray(t1_slice)
        t2_slice = np.asarray(t2_slice)
        
        # Resize if necessary
        if t1_slice.shape != self.target_size:
            t1_slice = torch.nn.functional.interpolate(
                torch.from_numpy(t1_slice).float().unsqueeze(0).unsqueeze(0),
                size=self.target_size,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        else:
            t1_slice = torch.from_numpy(t1_slice).float().unsqueeze(0)
            
        if t2_slice.shape != self.target_size:
            t2_slice = torch.nn.functional.interpolate(
                torch.from_numpy(t2_slice).float().unsqueeze(0).unsqueeze(0),
                size=self.target_size,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        else:
            t2_slice = torch.from_numpy(t2_slice).float().unsqueeze(0)
        
        # Apply normalization if requested
        if self.normalize:
            t1_slice = (t1_slice - t1_slice.min()) / (t1_slice.max() - t1_slice.min() + 1e-8)
            t2_slice = (t2_slice - t2_slice.min()) / (t2_slice.max() - t2_slice.min() + 1e-8)
        
        # Apply additional transforms if specified
        if self.transform:
            t1_slice = self.transform(t1_slice)
            t2_slice = self.transform(t2_slice)
        
        return t1_slice.contiguous(), t2_slice.contiguous()

def visualize_dataset_samples(dataset, num_samples=5, figsize=(20, 15)):
    """
    Visualize random samples from the dataset with enhanced alignment check
    
    Args:
        dataset: MRISliceDataset instance
        num_samples: Number of samples to visualize
        figsize: Figure size
    """
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    fig, axes = plt.subplots(4, num_samples, figsize=figsize)
    
    for i, idx in enumerate(indices):
        t1_slice, t2_slice = dataset[idx]
        
        # Convert to numpy and normalize for visualization if needed
        t1_np = t1_slice.squeeze().numpy()
        t2_np = t2_slice.squeeze().numpy()
        
        # Display T1 slice (transposed)
        axes[0, i].imshow(t1_np, cmap='gray')
        axes[0, i].set_title(f"T1 Slice (Transposed) {idx}")
        axes[0, i].axis('off')
        
        # Display T2 slice
        axes[1, i].imshow(t2_np, cmap='gray')
        axes[1, i].set_title(f"T2 Slice {idx}")
        axes[1, i].axis('off')
        
        # Display overlay of transposed T1 and T2
        axes[2, i].imshow(t1_np, cmap='gray')
        axes[2, i].imshow(t2_np, cmap='hot', alpha=0.5)
        axes[2, i].set_title("T1-T2 Overlay")
        axes[2, i].axis('off')
        
        # Display difference map
        diff = np.abs(t1_np - t2_np)
        diff_normalized = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
        axes[3, i].imshow(diff_normalized, cmap='viridis')
        axes[3, i].set_title("Difference Map")
        axes[3, i].axis('off')
    
    # Add row labels
    if num_samples > 0:
        axes[0, 0].set_ylabel("T1 Images", fontsize=12)
        axes[1, 0].set_ylabel("T2 Images\n(Transposed)", fontsize=12)
        axes[2, 0].set_ylabel("Overlay", fontsize=12)
        axes[3, 0].set_ylabel("Difference", fontsize=12)
    
    plt.tight_layout()
    plt.suptitle("T1-T2 Slice Correspondence Analysis", fontsize=14)
    plt.subplots_adjust(top=0.92)
    
    return fig

def create_data_loaders(t1_dir, t2_dir, batch_size=4, train_ratio=0.8, 
                       normalize=True, num_workers=4, transform=None):
    """
    Create train and validation data loaders
    
    Args:
        t1_dir: Directory containing T1 slices
        t2_dir: Directory containing T2 slices
        batch_size: Batch size
        train_ratio: Ratio of data to use for training
        normalize: Whether to normalize images
        num_workers: Number of workers for data loading
        transform: Optional transformations to apply
        
    Returns:
        train_loader, val_loader
    """
    # Create dataset
    dataset = MRISliceDataset(t1_dir, t2_dir, transform=transform, normalize=normalize)
    
    # Split into train and validation sets
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test MRI Slice Dataset")
    parser.add_argument("--t1_dir", type=str, default="./processed_dataset/IXI-T1", help="Directory containing T1 volumes")
    parser.add_argument("--t2_dir", type=str, default="./processed_dataset/IXI-T2", help="Directory containing T2 volumes")
    parser.add_argument("--visualize", action="store_true", help="Generate and save visualization")
    parser.add_argument("--slice_axis", type=int, default=2, help="Axis to extract slices from (0, 1, or 2)")
    args = parser.parse_args()
    
    t1_dir = args.t1_dir
    t2_dir = args.t2_dir
    
    # Simple test to ensure the dataset works correctly
    if os.path.exists(t1_dir) and os.path.exists(t2_dir):
        # Create dataset
        dataset = MRISliceDataset(t1_dir, t2_dir, slice_axis=args.slice_axis)
        
        # Test loading a sample
        if len(dataset) > 0:
            t1_slice, t2_slice = dataset[0]
            print(f"T1 slice shape: {t1_slice.shape}, T2 slice shape: {t2_slice.shape}")
            
            # Visualize samples
            if args.visualize:
                fig = visualize_dataset_samples(dataset)
                os.makedirs("./visualizations", exist_ok=True)
                fig.savefig("./visualizations/dataset_samples.png")
                print(f"Saved sample visualization to ./visualizations/dataset_samples.png")
        else:
            print("Dataset is empty. Please run process_data.py first.")
    else:
        print(f"Processed data directories not found: {t1_dir} or {t2_dir}")
        print("Please run process_data.py first.")