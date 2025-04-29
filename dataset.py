import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms
import argparse
import json
from pathlib import Path

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

def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, 
                  random_seed=42, save_name="./dataset_splits/dataset_split.json"):
    """
    Split dataset into train, validation and test sets
    
    Args:
        dataset: Dataset instance
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set  
        test_ratio: Ratio for test set
        random_seed: Random seed for reproducibility
        save_name: Name of the file to save the split indices
        
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    # Ensure ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Get total size and calculate split sizes
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    # Create random indices
    indices = np.random.permutation(total_size)
    
    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create directory to save indices
    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    
    # Save indices for reproducibility
    split_info = {
        'train_indices': train_indices.tolist(),
        'val_indices': val_indices.tolist(),
        'test_indices': test_indices.tolist(),
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'random_seed': random_seed,
        'total_size': total_size
    }
    
    with open(save_name, 'w') as f:
        json.dump(split_info, f)
    
    # Create subsets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    print(f"Dataset split: {len(train_dataset)} training, {len(val_dataset)} validation, {len(test_dataset)} test samples")
    print(f"Split indices saved to {save_name}")
    
    return train_dataset, val_dataset, test_dataset

def load_existing_split(dataset, split_file):
    """
    Load existing dataset split from a file
    
    Args:
        dataset: Dataset instance
        split_file: Path to the split JSON file
        
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")
    
    with open(split_file, 'r') as f:
        split_info = json.load(f)
    
    train_indices = split_info['train_indices']
    val_indices = split_info['val_indices']
    test_indices = split_info['test_indices']
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    print(f"Loaded dataset split: {len(train_dataset)} training, {len(val_dataset)} validation, {len(test_dataset)} test samples")
    return train_dataset, val_dataset, test_dataset

def create_data_loaders(t1_dir, t2_dir, batch_size=4, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                       normalize=True, num_workers=4, transform=None, random_seed=42,
                       split_dir=None, use_existing_split=True):
    """
    Create train, validation, and test data loaders with proper splitting
    
    Args:
        t1_dir: Directory containing T1 slices
        t2_dir: Directory containing T2 slices
        batch_size: Batch size
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        normalize: Whether to normalize images
        num_workers: Number of workers for data loading
        transform: Optional transformations to apply
        random_seed: Random seed for reproducibility
        split_dir: Directory to save/load split information
        use_existing_split: Whether to use existing split if available
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create dataset
    dataset = MRISliceDataset(t1_dir, t2_dir, transform=transform, normalize=normalize)
    
    if split_dir is None:
        split_dir = f"./{os.path.basename(t1_dir)}_{os.path.basename(t2_dir)}_splits"

    # print(f"Split directory: {split_dir}")

    # Check if we should use existing split
    split_file = os.path.join(split_dir, f'{train_ratio}_{val_ratio}_{test_ratio}_{random_seed}.json')
    if use_existing_split and os.path.exists(split_file):
        print("Using existing split")
        train_dataset, val_dataset, test_dataset = load_existing_split(dataset, split_file)
    else:
        print("Creating new split")
        # Create a new split
        train_dataset, val_dataset, test_dataset = split_dataset(
            dataset, 
            train_ratio=train_ratio, 
            val_ratio=val_ratio, 
            test_ratio=test_ratio,
            random_seed=random_seed,
            save_name=split_file
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
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test MRI Slice Dataset")
    parser.add_argument("--t1_dir", type=str, default="./processed_dataset/IXI-T1", help="Directory containing T1 volumes")
    parser.add_argument("--t2_dir", type=str, default="./processed_dataset/IXI-T2", help="Directory containing T2 volumes")
    parser.add_argument("--visualize", action="store_true", help="Generate and save visualization")
    parser.add_argument("--slice_axis", type=int, default=2, help="Axis to extract slices from (0, 1, or 2)")
    parser.add_argument("--create_splits", action="store_true", help="Create and save train/val/test splits")
    parser.add_argument("--split_dir", type=str, default="./dataset_splits", help="Directory to save split information")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Ratio of data for training")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Ratio of data for validation")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Ratio of data for testing")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for dataset splitting")
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
            
            # Create train/val/test splits if requested
            if args.create_splits:
                train_dataset, val_dataset, test_dataset = split_dataset(
                    dataset,
                    train_ratio=args.train_ratio,
                    val_ratio=args.val_ratio,
                    test_ratio=args.test_ratio,
                    random_seed=args.random_seed,
                    save_dir=args.split_dir
                )
                
                # Visualize some examples from each split
                if args.visualize:
                    splits = {
                        'train': train_dataset, 
                        'validation': val_dataset,
                        'test': test_dataset
                    }
                    
                    os.makedirs("./visualizations", exist_ok=True)
                    
                    for split_name, split_dataset in splits.items():
                        if len(split_dataset) > 0:
                            fig = visualize_dataset_samples(split_dataset, num_samples=min(3, len(split_dataset)))
                            fig.savefig(f"./visualizations/{split_name}_samples.png")
                            print(f"Saved {split_name} visualization to ./visualizations/{split_name}_samples.png")
            
            # Visualize general samples if requested
            elif args.visualize:
                fig = visualize_dataset_samples(dataset)
                os.makedirs("./visualizations", exist_ok=True)
                fig.savefig("./visualizations/dataset_samples.png")
                print(f"Saved sample visualization to ./visualizations/dataset_samples.png")
        else:
            print("Dataset is empty. Please run process_data.py first.")
    else:
        print(f"Processed data directories not found: {t1_dir} or {t2_dir}")
        print("Please run process_data.py first.")