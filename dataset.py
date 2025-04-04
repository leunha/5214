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
    Dataset for paired 2D MRI slices (T1 and T2)
    """
    def __init__(self, t1_dir, t2_dir, transform=None, normalize=True, slice_axis=2):
        """
        Initialize the dataset.
        
        Args:
            t1_dir: Directory containing T1 volumes
            t2_dir: Directory containing T2 volumes
            transform: Optional transformations to apply
            normalize: Whether to normalize images to [0, 1]
            slice_axis: Axis to slice along (default: 2 for axial slices)
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
        
        # Load all volumes to get slice counts
        self.slices_per_volume = []
        self.total_slices = 0
        
        for i in range(len(self.t1_files)):
            t1_volume = np.load(self.t1_files[i])
            t2_volume = np.load(self.t2_files[i])
            
            # Get the minimum slice count between T1 and T2
            min_slices = min(t1_volume.shape[self.slice_axis], t2_volume.shape[self.slice_axis])
            self.slices_per_volume.append(min_slices)
            self.total_slices += min_slices
        
        print(f"Dataset contains {len(self.t1_files)} paired volumes with a total of {self.total_slices} paired slices")
    
    def __len__(self):
        return self.total_slices
    
    def __getitem__(self, idx):
        # Find which volume and slice this index corresponds to
        volume_idx = 0
        slice_idx = idx
        
        while slice_idx >= self.slices_per_volume[volume_idx]:
            slice_idx -= self.slices_per_volume[volume_idx]
            volume_idx += 1
        
        # Load the T1 and T2 volumes
        t1_volume = np.load(self.t1_files[volume_idx])
        t2_volume = np.load(self.t2_files[volume_idx])
        
        # Extract the slices
        if self.slice_axis == 0:
            t1_slice = t1_volume[slice_idx, :, :]
            t2_slice = t2_volume[slice_idx, :, :]
        elif self.slice_axis == 1:
            t1_slice = t1_volume[:, slice_idx, :]
            t2_slice = t2_volume[:, slice_idx, :]
        else:  # self.slice_axis == 2
            t1_slice = t1_volume[:, :, slice_idx]
            t2_slice = t2_volume[:, :, slice_idx]
        
        # Convert to tensors
        t1_tensor = torch.from_numpy(t1_slice).float().unsqueeze(0)  # Add channel dimension
        t2_tensor = torch.from_numpy(t2_slice).float().unsqueeze(0)  # Add channel dimension
        
        # Apply optional normalization to [0, 1]
        if self.normalize:
            t1_tensor = (t1_tensor - t1_tensor.min()) / (t1_tensor.max() - t1_tensor.min() + 1e-8)
            t2_tensor = (t2_tensor - t2_tensor.min()) / (t2_tensor.max() - t2_tensor.min() + 1e-8)
        
        # Apply transformations if specified
        if self.transform:
            t1_tensor = self.transform(t1_tensor)
            t2_tensor = self.transform(t2_tensor)
        
        return t1_tensor, t2_tensor

def visualize_dataset_samples(dataset, num_samples=5, figsize=(15, 8)):
    """
    Visualize random samples from the dataset
    
    Args:
        dataset: MRISliceDataset instance
        num_samples: Number of samples to visualize
        figsize: Figure size
    """
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    fig, axes = plt.subplots(2, num_samples, figsize=figsize)
    
    for i, idx in enumerate(indices):
        t1_slice, t2_slice = dataset[idx]
        
        # Display T1 and T2 slices
        axes[0, i].imshow(t1_slice.squeeze().numpy(), cmap='gray')
        axes[0, i].set_title(f"T1 Slice {idx}")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(t2_slice.squeeze().numpy(), cmap='gray')
        axes[1, i].set_title(f"T2 Slice {idx}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
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