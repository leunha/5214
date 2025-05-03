import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import pandas as pd
import json
from tqdm import tqdm
import time
from monai.metrics import MAEMetric, MSEMetric

from dataset import MRISliceDataset, create_data_loaders, load_existing_split
from monai_rectified_flow import MonaiRectifiedFlow, RectifiedFlowODE

def calculate_metrics(source, generated, target):
    """
    Calculate various image quality metrics between generated and target images
    
    Args:
        source: Source image (T1)
        generated: Generated image (predicted T2)
        target: Target image (true T2)
        
    Returns:
        dict: Dictionary containing the computed metrics
    """
    # Convert to numpy arrays and ensure they are in the right shape
    if torch.is_tensor(source):
        source = source.squeeze().cpu().numpy()
    if torch.is_tensor(generated):
        generated = generated.squeeze().cpu().numpy()
    if torch.is_tensor(target):
        target = target.squeeze().cpu().numpy()
    
    # Ensure values are in [0, 1] range
    source = np.clip(source, 0, 1)
    generated = np.clip(generated, 0, 1)
    target = np.clip(target, 0, 1)
    
    # Calculate PSNR
    psnr_value = psnr(target, generated, data_range=1.0)
    
    # Calculate SSIM
    ssim_value = ssim(target, generated, data_range=1.0)
    
    # Calculate Mean Squared Error (MSE)
    mse_value = np.mean((target - generated) ** 2)
    
    # Calculate Mean Absolute Error (MAE)
    mae_value = np.mean(np.abs(target - generated))
    
    return {
        'psnr': float(psnr_value),
        'ssim': float(ssim_value),
        'mse': float(mse_value),
        'mae': float(mae_value)
    }

def visualize_results(source_batch, generated_batch, target_batch, trajectories, output_dir, prefix='', num_steps=50, reflow=0):
    """
    Visualize evaluation results
    
    Args:
        source_batch: Batch of source images (T1)
        generated_batch: Batch of generated images (T2)
        target_batch: Batch of target images (true T2)
        trajectories: List of trajectory states
        output_dir: Directory to save visualizations
        prefix: Prefix for output filenames
        num_steps: Number of steps for ODE integration
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize sample results
    n_samples = min(4, source_batch.size(0))
    
    # Create figure
    fig, axes = plt.subplots(3, n_samples, figsize=(4*n_samples, 10))
    
    # For each sample
    for i in range(n_samples):
        # Show source (T1)
        axes[0, i].imshow(source_batch[i, 0].cpu().numpy(), cmap='gray')
        axes[0, i].set_title('Source (T1)')
        axes[0, i].axis('off')
        
        # Show generated (predicted T2)
        axes[1, i].imshow(generated_batch[i, 0].cpu().numpy(), cmap='gray')
        axes[1, i].set_title('Generated (T2)')
        axes[1, i].axis('off')
        
        # Show target (true T2)
        axes[2, i].imshow(target_batch[i, 0].cpu().numpy(), cmap='gray')
        axes[2, i].set_title('Target (True T2)')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}samples_{num_steps}_reflow_{reflow}.png'))
    plt.close()
    
    # Visualize trajectory for one sample
    fig, axes = plt.subplots(1, min(8, len(trajectories)), figsize=(20, 4))
    
    # Select a subset of time steps
    step_indices = np.linspace(0, len(trajectories) - 1, min(8, len(trajectories))).astype(int)
    
    for i, idx in enumerate(step_indices):
        if len(step_indices) == 1:  # Handle the case of just a single subplot
            ax = axes
        else:
            ax = axes[i]
            
        ax.imshow(trajectories[idx][0, 0].cpu().numpy(), cmap='gray')
        ax.set_title(f'Step {idx}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}trajectory_{num_steps}_reflow_{reflow}.png'))
    plt.close()
    
    # Visualize difference maps
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))
    
    for i in range(n_samples):
        # Original image
        axes[i, 0].imshow(target_batch[i, 0].cpu().numpy(), cmap='gray')
        axes[i, 0].set_title('Target (True T2)')
        axes[i, 0].axis('off')
        
        # Generated image
        axes[i, 1].imshow(generated_batch[i, 0].cpu().numpy(), cmap='gray')
        axes[i, 1].set_title('Generated (T2)')
        axes[i, 1].axis('off')
        
        # Difference map
        diff = np.abs(target_batch[i, 0].cpu().numpy() - generated_batch[i, 0].cpu().numpy())
        im = axes[i, 2].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
        axes[i, 2].set_title('Difference Map')
        axes[i, 2].axis('off')
        
        # Add colorbar
        if i == 0:
            plt.colorbar(im, ax=axes[i, 2])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}difference_maps_{num_steps}_reflow_{reflow}.png'))
    plt.close()

def evaluate_model(model, test_loader, device, num_steps=50, output_dir='./evaluation_results', test_samples=None, reflow=0):
    """
    Evaluate the model on test data
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device: Device to use
        num_steps: Number of steps for ODE integration
        output_dir: Directory to save results
        test_samples: Number of samples to test (None = all)
        reflow: Number of reflow steps

    Returns:
        pd.DataFrame: Metrics for each test sample
    """
    model.eval()
    rf = RectifiedFlowODE(model, num_steps=num_steps)
    
    all_metrics = []
    visualization_done = False
    
    with torch.no_grad():
        for i, (source_batch, target_batch) in enumerate(tqdm(test_loader, desc="Evaluating")):
            if test_samples is not None and i >= test_samples:
                break
                
            source_batch, target_batch = source_batch.to(device), target_batch.to(device)
            
            # Time the generation process
            start_time = time.time()
            trajectories = rf.sample_ode(source_batch, N=num_steps)
            generated_batch = trajectories[-1]
            generation_time = time.time() - start_time
            
            # Calculate metrics for each image in the batch
            batch_metrics = []
            for j in range(source_batch.size(0)):
                source = source_batch[j]
                generated = generated_batch[j]
                target = target_batch[j]
                
                metrics = calculate_metrics(source, generated, target)
                metrics['sample_idx'] = i * test_loader.batch_size + j
                metrics['generation_time'] = generation_time / source_batch.size(0)
                
                batch_metrics.append(metrics)
            
            all_metrics.extend(batch_metrics)
            
            # Visualize the first batch
            if not visualization_done:
                visualize_results(
                    source_batch, 
                    generated_batch, 
                    target_batch, 
                    trajectories, 
                    output_dir,
                    num_steps=num_steps,
                    reflow=reflow
                )
                visualization_done = True
    
    # Convert metrics to DataFrame
    df_metrics = pd.DataFrame(all_metrics)
    
    # Save metrics CSV
    os.makedirs(output_dir, exist_ok=True)
    df_metrics.to_csv(os.path.join(output_dir, f'metrics_{num_steps}_reflow_{reflow}.csv'), index=False)
    
    # Calculate and print summary statistics
    summary = df_metrics.describe()
    summary.to_csv(os.path.join(output_dir, f'metrics_summary_{num_steps}_reflow_{reflow}.csv'))
    
    print("\nMetrics Summary:")
    print(summary)
    
    # Create metrics visualizations
    plt.figure(figsize=(10, 6))
    plt.boxplot([df_metrics['psnr'], df_metrics['ssim'], df_metrics['mse'] * 100, df_metrics['mae'] * 100])
    plt.xticks([1, 2, 3, 4], ['PSNR', 'SSIM', 'MSE×100', 'MAE×100'])
    plt.title('Metrics Distribution')
    plt.savefig(os.path.join(output_dir, f'metrics_boxplot_{num_steps}_reflow_{reflow}.png'))
    plt.close()
    
    # Create histograms for each metric
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    axes[0].hist(df_metrics['psnr'], bins=20)
    axes[0].set_title('PSNR Distribution')
    axes[0].set_xlabel('PSNR (dB)')
    
    axes[1].hist(df_metrics['ssim'], bins=20)
    axes[1].set_title('SSIM Distribution')
    axes[1].set_xlabel('SSIM')
    
    axes[2].hist(df_metrics['mse'], bins=20)
    axes[2].set_title('MSE Distribution')
    axes[2].set_xlabel('MSE')
    
    axes[3].hist(df_metrics['mae'], bins=20)
    axes[3].set_title('MAE Distribution')
    axes[3].set_xlabel('MAE')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'metrics_histograms_{num_steps}_reflow_{reflow}.png'))
    plt.close()
    
    return df_metrics

def main(args):
    # Set device
    if args.device is None:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)
    
    print(f"Using device: {args.device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(args.output_dir, 'eval_config.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    # Check if we should use a specific split file
    if args.split_file and os.path.exists(args.split_file):
        print(f"Loading test set from split file: {args.split_file}")
        # Create the full dataset
        full_dataset = MRISliceDataset(args.t1_dir, args.t2_dir)
        
        # Load the existing split but only use the test set
        _, _, test_dataset = load_existing_split(full_dataset, args.split_file)
        
        print(f"Using dedicated test set with {len(test_dataset)} samples")
    else:
        print("No valid split file specified, creating dataset from scratch")
        # Create dataset
        test_dataset = MRISliceDataset(args.t1_dir, args.t2_dir)
        
        # Use a subset for testing if specified
        if args.test_subset > 0 and args.test_subset < len(test_dataset):
            print(f"WARNING: Randomly sampling {args.test_subset} images for testing.")
            print("This approach may include images seen during training and is not recommended.")
            print("Use the --split_file option to use a proper train/val/test split.")
            indices = np.random.choice(len(test_dataset), args.test_subset, replace=False)
            test_dataset = torch.utils.data.Subset(test_dataset, indices)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    print(f"Testing on {len(test_dataset)} samples")
    
    # Load model
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model_args = checkpoint.get('args', {})
    
    # Get image size from the first batch
    sample_batch = next(iter(test_loader))
    img_size = sample_batch[0].shape[2]
    
    # Create model with the same configuration as training
    features = model_args.get('features', [32, 64, 128, 256])  # Use the features from training
    print(f"Creating model with features: {features}")
    
    model = MonaiRectifiedFlow(
        img_size=img_size,
        in_channels=1,
        out_channels=1,
        features=features
    ).to(args.device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {args.model_path}")
    
    # Evaluate
    metrics = evaluate_model(
        model,
        test_loader,
        args.device,
        num_steps=args.num_steps,
        output_dir=args.output_dir,
        test_samples=args.max_test_batches
    )
    
    # Check for reflowed model
    reflowed_path = None
    model_dir = os.path.dirname(args.model_path)
    
    # Try standard naming convention
    standard_reflowed_path = args.model_path.replace('final_model.pt', 'reflowed_model.pt')
    if os.path.exists(standard_reflowed_path):
        reflowed_path = standard_reflowed_path
    # Try looking in the same directory
    elif os.path.exists(os.path.join(model_dir, 'reflowed_model.pt')):
        reflowed_path = os.path.join(model_dir, 'reflowed_model.pt')
    
    # Only evaluate reflowed model if we actually found one
    if reflowed_path and os.path.exists(reflowed_path) and reflowed_path != args.model_path:
        print(f"\nEvaluating reflowed model: {reflowed_path}")
        
        reflowed_checkpoint = torch.load(reflowed_path, map_location=args.device)
        reflowed_model = MonaiRectifiedFlow(
            img_size=img_size,
            in_channels=1,
            out_channels=1,
            features=features
        ).to(args.device)
        
        reflowed_model.load_state_dict(reflowed_checkpoint['model_state_dict'])
        
        reflowed_metrics = evaluate_model(
            reflowed_model,
            test_loader,
            args.device,
            num_steps=args.num_steps,
            output_dir=os.path.join(args.output_dir, 'reflowed'),
            test_samples=args.max_test_batches
        )
        
        # Compare original vs reflowed
        print("\nComparison: Original vs Reflowed")
        comparison = pd.DataFrame({
            'Metric': ['PSNR', 'SSIM', 'MSE', 'MAE'],
            'Original': [
                metrics['psnr'].mean(),
                metrics['ssim'].mean(),
                metrics['mse'].mean(),
                metrics['mae'].mean()
            ],
            'Reflowed': [
                reflowed_metrics['psnr'].mean(),
                reflowed_metrics['ssim'].mean(),
                reflowed_metrics['mse'].mean(),
                reflowed_metrics['mae'].mean()
            ]
        })
        
        comparison.to_csv(os.path.join(args.output_dir, 'comparison.csv'), index=False)
        print(comparison)
        
        # Create bar chart comparison
        plt.figure(figsize=(10, 6))
        x = np.arange(4)
        width = 0.35
        
        plt.bar(x - width/2, comparison['Original'], width, label='Original')
        plt.bar(x + width/2, comparison['Reflowed'], width, label='Reflowed')
        
        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.title('Original vs Reflowed Model')
        plt.xticks(x, comparison['Metric'])
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'comparison.png'))
        plt.close()
    else:
        print("\nNo valid reflowed model found. Skipping comparison.")
    
    print(f"Evaluation results saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate MONAI Rectified Flow for MRI Translation')
    
    # Data arguments
    parser.add_argument('--t1_dir', type=str, required=True, help='Directory containing T1 slices')
    parser.add_argument('--t2_dir', type=str, required=True, help='Directory containing T2 slices')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    
    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda or cpu)')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results', help='Directory to save results')
    parser.add_argument('--num_steps', type=int, default=50, help='Number of steps for ODE solution')
    
    # Dataset arguments - legacy support
    parser.add_argument('--test_subset', type=int, default=0, help='Number of samples to test (legacy, use --split_file instead)')
    parser.add_argument('--max_test_batches', type=int, default=None, help='Maximum number of batches to test')
    
    # New split file argument
    parser.add_argument('--split_file', type=str, default='./dataset_splits/dataset_split.json', 
                        help='Path to dataset split JSON file (to use the dedicated test set)')
    
    args = parser.parse_args()
    main(args) 