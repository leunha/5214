import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import time

from rectified_flow import ImageRectifiedFlow, RectifiedFlow, train_rectified_flow

class MRIDataset(Dataset):
    """
    Dataset for paired MRI images (T1 and T2)
    """
    def __init__(self, t1_dir, t2_dir, transform=None):
        """
        Args:
            t1_dir: Directory with T1 images
            t2_dir: Directory with T2 images
            transform: Optional transform to apply to the data
        """
        self.t1_files = sorted([os.path.join(t1_dir, f) for f in os.listdir(t1_dir) if f.endswith('.npy')])
        self.t2_files = sorted([os.path.join(t2_dir, f) for f in os.listdir(t2_dir) if f.endswith('.npy')])
        
        # Ensure we have matching T1 and T2 files
        t1_subjects = [os.path.basename(f).split('-')[1] for f in self.t1_files]
        t2_subjects = [os.path.basename(f).split('-')[1] for f in self.t2_files]
        
        # Find common subjects
        common_subjects = set(t1_subjects).intersection(set(t2_subjects))
        
        # Filter files to only include common subjects
        self.t1_files = [f for f in self.t1_files if os.path.basename(f).split('-')[1] in common_subjects]
        self.t2_files = [f for f in self.t2_files if os.path.basename(f).split('-')[1] in common_subjects]
        
        # Sort to ensure corresponding pairs
        self.t1_files.sort()
        self.t2_files.sort()
        
        self.transform = transform
        
        print(f"Dataset contains {len(self.t1_files)} paired samples")
    
    def __len__(self):
        return len(self.t1_files)
    
    def __getitem__(self, idx):
        # Load the T1 and T2 images
        t1_img = np.load(self.t1_files[idx])
        t2_img = np.load(self.t2_files[idx])
        
        # Convert to tensors
        t1_tensor = torch.from_numpy(t1_img).float()
        t2_tensor = torch.from_numpy(t2_img).float()
        
        # Handle the case where T1 has more slices than T2
        if len(t1_tensor.shape) == 3 and len(t2_tensor.shape) == 3:
            t1_slices = t1_tensor.shape[2]
            t2_slices = t2_tensor.shape[2]
            
            # Check if T1 has more slices than T2
            if t1_slices > t2_slices:
                # Calculate offset to center the T1 slices relative to T2
                # This assumes the extra slices in T1 are distributed evenly at the beginning and end
                offset = (t1_slices - t2_slices) // 2
                
                # If T1 has exactly 20 more slices, use this specific offset
                if t1_slices - t2_slices == 20:
                    offset = 10  # Half of the 20 extra slices
                
                # Select a random slice from T2's range
                t2_slice_idx = np.random.randint(0, t2_slices)
                # Map to corresponding T1 slice with offset
                t1_slice_idx = t2_slice_idx + offset
                
                # Safety check to ensure indices are valid
                t1_slice_idx = min(max(0, t1_slice_idx), t1_slices - 1)
                t2_slice_idx = min(max(0, t2_slice_idx), t2_slices - 1)
                
                # Extract slices
                t1_slice = t1_tensor[:, :, t1_slice_idx]
                t2_slice = t2_tensor[:, :, t2_slice_idx]
            else:
                # If T1 and T2 have same number of slices or T2 has more (unexpected case)
                # Pick a common range and select a random slice
                common_slices = min(t1_slices, t2_slices)
                slice_idx = np.random.randint(0, common_slices)
                t1_slice = t1_tensor[:, :, slice_idx]
                t2_slice = t2_tensor[:, :, slice_idx]
        else:
            # Handle the case where one or both are not 3D
            t1_slice = t1_tensor
            t2_slice = t2_tensor
        
        # Normalize to [0, 1]
        t1_slice = (t1_slice - t1_slice.min()) / (t1_slice.max() - t1_slice.min() + 1e-8)
        t2_slice = (t2_slice - t2_slice.min()) / (t2_slice.max() - t2_slice.min() + 1e-8)
        
        # Add channel dimension
        t1_slice = t1_slice.unsqueeze(0)
        t2_slice = t2_slice.unsqueeze(0)
        
        # Ensure consistent size (e.g., 256x256)
        if t1_slice.shape[1] != 256 or t1_slice.shape[2] != 256:
            t1_slice = torch.nn.functional.interpolate(t1_slice.unsqueeze(0), size=(256, 256), mode='bilinear').squeeze(0)
            t2_slice = torch.nn.functional.interpolate(t2_slice.unsqueeze(0), size=(256, 256), mode='bilinear').squeeze(0)
        
        if self.transform:
            t1_slice = self.transform(t1_slice)
            t2_slice = self.transform(t2_slice)
        
        return t1_slice, t2_slice

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

def main():
    pass
#write the main function later
if __name__ == "__main__":
    main()