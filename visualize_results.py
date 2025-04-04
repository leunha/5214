import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import MRISliceDataset
from monai_rectified_flow import MonaiRectifiedFlow, RectifiedFlowODE

def visualize_t1_to_t2_transformation(model_path, t1_dir, t2_dir, output_dir='./visualizations', device='cpu', sample_indices=None):
    """
    Visualize T1 to T2 transformation using the trained model.
    
    Args:
        model_path: Path to the trained model checkpoint
        t1_dir: Directory containing T1 volumes
        t2_dir: Directory containing T2 volumes
        output_dir: Directory to save visualizations
        device: Device to use for inference
        sample_indices: List of sample indices to visualize (if None, random samples are chosen)
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device)
    
    # Load dataset
    dataset = MRISliceDataset(t1_dir, t2_dir)
    
    # Choose sample indices
    if sample_indices is None:
        sample_indices = np.random.choice(len(dataset), 5, replace=False)
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model_args = checkpoint.get('args', {})
    
    # Get features from checkpoint
    features = model_args.get('features', [32, 64, 128])
    
    # Create model
    sample_t1, _ = dataset[0]
    img_size = sample_t1.shape[1]  # Get image size from first sample
    
    model = MonaiRectifiedFlow(
        img_size=img_size,
        in_channels=1,
        out_channels=1,
        features=features
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create ODE solver
    rf = RectifiedFlowODE(model, num_steps=20)
    
    # Create figure for results
    fig, axes = plt.subplots(4, len(sample_indices), figsize=(4*len(sample_indices), 12))
    
    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            # Get sample pair
            t1_slice, t2_slice = dataset[idx]
            
            # Move to device
            t1_slice = t1_slice.unsqueeze(0).to(device)  # Add batch dimension
            t2_slice = t2_slice.unsqueeze(0).to(device)  # Add batch dimension
            
            # Generate T2 from T1
            trajectories = rf.sample_ode(t1_slice, N=20)
            generated_t2 = trajectories[-1]
            
            # Visualize
            # Row 1: T1 slice
            axes[0, i].imshow(t1_slice[0, 0].cpu().numpy(), cmap='gray')
            axes[0, i].set_title(f"T1 Input (Sample {idx})")
            axes[0, i].axis('off')
            
            # Row 2: Generated T2 slice
            axes[1, i].imshow(generated_t2[0, 0].cpu().numpy(), cmap='gray')
            axes[1, i].set_title("Generated T2")
            axes[1, i].axis('off')
            
            # Row 3: Ground truth T2 slice
            axes[2, i].imshow(t2_slice[0, 0].cpu().numpy(), cmap='gray')
            axes[2, i].set_title("Ground Truth T2")
            axes[2, i].axis('off')
            
            # Row 4: Overlay of generated and ground truth
            axes[3, i].imshow(t2_slice[0, 0].cpu().numpy(), cmap='gray')
            axes[3, i].imshow(generated_t2[0, 0].cpu().numpy(), cmap='hot', alpha=0.5)
            axes[3, i].set_title("Overlay")
            axes[3, i].axis('off')
    
    # Add row labels
    if len(sample_indices) > 0:
        axes[0, 0].set_ylabel("T1 Input", fontsize=12)
        axes[1, 0].set_ylabel("Generated T2", fontsize=12)
        axes[2, 0].set_ylabel("Ground Truth T2", fontsize=12)
        axes[3, 0].set_ylabel("Overlay", fontsize=12)
    
    plt.tight_layout()
    plt.suptitle("T1 to T2 MRI Translation", fontsize=16)
    plt.subplots_adjust(top=0.94)
    
    # Save figure
    output_path = os.path.join(output_dir, "t1_to_t2_results.png")
    plt.savefig(output_path)
    print(f"Results saved to {output_path}")
    
    # Create figure showing transformation trajectory
    plt.figure(figsize=(20, 4))
    num_steps = min(8, len(trajectories))
    step_indices = np.linspace(0, len(trajectories) - 1, num_steps).astype(int)
    
    for i, step_idx in enumerate(step_indices):
        plt.subplot(1, num_steps, i + 1)
        plt.imshow(trajectories[step_idx][0, 0].cpu().numpy(), cmap='gray')
        plt.title(f"Step {step_idx}")
        plt.axis('off')
    
    plt.suptitle("T1 to T2 Transformation Trajectory", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save trajectory figure
    trajectory_path = os.path.join(output_dir, "t1_to_t2_trajectory.png")
    plt.savefig(trajectory_path)
    print(f"Trajectory saved to {trajectory_path}")
    
    return output_path, trajectory_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize T1 to T2 Transformation')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--t1_dir', type=str, default='./processed_dataset/IXI-T1', help='Directory with T1 volumes')
    parser.add_argument('--t2_dir', type=str, default='./processed_dataset/IXI-T2', help='Directory with T2 volumes')
    parser.add_argument('--output_dir', type=str, default='./visualizations', help='Output directory for visualizations')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu or cuda)')
    parser.add_argument('--sample_idx', type=int, nargs='+', help='Specific sample indices to visualize')
    
    args = parser.parse_args()
    
    visualize_t1_to_t2_transformation(
        args.model_path,
        args.t1_dir,
        args.t2_dir,
        args.output_dir,
        args.device,
        args.sample_idx
    ) 