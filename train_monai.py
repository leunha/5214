import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from datetime import datetime
import torch.nn.functional as F

from dataset import MRISliceDataset, create_data_loaders
from monai_rectified_flow import MonaiRectifiedFlow, RectifiedFlowODE, train_monai_rectified_flow

def save_checkpoint(model, optimizer, epoch, loss, args):
    """Save model checkpoint"""
    os.makedirs(args.output_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'args': vars(args)
    }
    
    checkpoint_path = os.path.join(args.output_dir, f'model_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

def visualize_results(model, rf, val_loader, epoch, args):
    """Visualize sample results during training"""
    model.eval()
    
    # Create output directory for visualizations
    os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)
    
    with torch.no_grad():
        # Get one batch from validation set
        source_batch, target_batch = next(iter(val_loader))
        source_batch, target_batch = source_batch.to(args.device), target_batch.to(args.device)
        
        # Select just a few samples to visualize
        source = source_batch[:4]
        target = target_batch[:4]
        
        # Generate trajectory
        trajectories = rf.sample_ode(source, N=args.num_steps)
        generated = trajectories[-1]
        
        # Create figure
        n_samples = min(4, source.size(0))
        fig, axes = plt.subplots(4, n_samples, figsize=(4*n_samples, 12))
        
        # For each sample
        for i in range(n_samples):
            # Show source (T1)
            axes[0, i].imshow(source[i, 0].cpu().numpy(), cmap='gray')
            axes[0, i].set_title('Source (T1)')
            axes[0, i].axis('off')
            
            # Show generated (predicted T2)
            axes[1, i].imshow(generated[i, 0].cpu().numpy(), cmap='gray')
            axes[1, i].set_title('Generated (T2)')
            axes[1, i].axis('off')
            
            # Show target (true T2)
            axes[2, i].imshow(target[i, 0].cpu().numpy(), cmap='gray')
            axes[2, i].set_title('Target (True T2)')
            axes[2, i].axis('off')
            
            # Show overlay of target and generated for alignment check
            axes[3, i].imshow(target[i, 0].cpu().numpy(), cmap='gray')
            axes[3, i].imshow(generated[i, 0].cpu().numpy(), cmap='hot', alpha=0.5)
            axes[3, i].set_title('True/Generated Overlay')
            axes[3, i].axis('off')
        
        # Add row labels
        if n_samples > 0:
            axes[0, 0].set_ylabel("Source (T1)", fontsize=12)
            axes[1, 0].set_ylabel("Generated (T2)", fontsize=12)
            axes[2, 0].set_ylabel("Target (T2)", fontsize=12)
            axes[3, 0].set_ylabel("Alignment Check", fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'visualizations', f'epoch_{epoch}.png'))
        plt.close()
        
        # Also create a trajectory visualization for one sample
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
        plt.savefig(os.path.join(args.output_dir, 'visualizations', f'trajectory_epoch_{epoch}.png'))
        plt.close()

def train_monai_rectified_flow(model, optimizer, source_loader, target_loader, device, epochs, output_dir):
    """
    Train the Rectified Flow model.
    
    Args:
        model: MonaiRectifiedFlow model
        optimizer: Optimizer
        source_loader: DataLoader for source domain (T1)
        target_loader: DataLoader for target domain (T2)
        device: Device for training
        epochs: Number of epochs
        output_dir: Directory to save checkpoints and visualizations
        
    Returns:
        model: Trained model
        loss_curve: Training loss history
    """
    model.train()
    loss_curve = []
    rf = RectifiedFlowODE(model, num_steps=20)  # Create RectifiedFlowODE instance for visualization
    
    for epoch in range(epochs):
        epoch_losses = []
        
        for batch_data in tqdm(source_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            # Get source and target images from the same batch
            source_batch, target_batch = batch_data
            
            source_batch = source_batch.to(device)
            target_batch = target_batch.to(device)
            
            # Ensure we have matching batch sizes
            current_batch_size = source_batch.size(0)
            
            # Sample random timesteps
            t = torch.rand(current_batch_size, device=device)
            
            # Interpolate between source and target
            z_t = source_batch * (1-t.view(-1,1,1,1)) + target_batch * t.view(-1,1,1,1)
            
            # Compute model prediction
            pred = model(z_t, t)
            
            # Compute loss (target - source is ground truth for rectified flow)
            loss = F.mse_loss(pred, target_batch - source_batch)
            
            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        loss_curve.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        # Save checkpoint after each epoch
        checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pt')
        try:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        except Exception as e:
            print(f"Error saving checkpoint: {str(e)}")
        
        # Generate and save visualizations
        try:
            model.eval()
            with torch.no_grad():
                # Get validation batch
                val_batch = next(iter(target_loader))
                source_batch, target_batch = val_batch
                source_batch, target_batch = source_batch.to(device), target_batch.to(device)
                
                # Generate trajectory
                trajectories = rf.sample_ode(source_batch[:4], N=20)
                generated = trajectories[-1]
                
                # Save visualization
                vis_path = os.path.join(output_dir, 'visualizations', f'results_epoch_{epoch+1}.png')
                os.makedirs(os.path.dirname(vis_path), exist_ok=True)
                
                # Create a namespace with both output_dir and device attributes
                args_namespace = argparse.Namespace(output_dir=output_dir, device=device, num_steps=20)
                
                visualize_results(model, rf, target_loader, epoch + 1, args_namespace)
                print(f"Saved visualization to {vis_path}")
            model.train()
        except Exception as e:
            print(f"Error generating visualization: {str(e)}")
            import traceback
            traceback.print_exc()  # Print full error traceback
    
    return model, loss_curve

def main(args):
    # Set device
    if args.device is None:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)
    
    print(f"Using device: {args.device}")
    
    # Set up output directory with absolute path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.abspath(os.path.join(args.output_dir, f"run_{timestamp}"))
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Create visualizations directory
    vis_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    print(f"Visualization directory: {vis_dir}")
    
    # Save configuration
    config_path = os.path.join(args.output_dir, 'config.txt')
    with open(config_path, 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    print(f"Saved configuration to {config_path}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        args.t1_dir,
        args.t2_dir,
        batch_size=args.batch_size,
        train_ratio=0.8,
        num_workers=args.num_workers
    )
    
    print(f"Training on {len(train_loader.dataset)} samples")
    print(f"Validating on {len(val_loader.dataset)} samples")
    
    # Check batch data shapes
    sample_batch = next(iter(train_loader))
    source_batch, target_batch = sample_batch
    print(f"Source shape: {source_batch.shape}, Target shape: {target_batch.shape}")
    
    # Create model
    model = MonaiRectifiedFlow(
        img_size=source_batch.shape[2],  # Use actual image size from the data
        in_channels=1,
        out_channels=1,
        features=args.features
    ).to(args.device)
    
    # Print model summary
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Train the model
    start_time = time.time()
    
    model, loss_curve = train_monai_rectified_flow(
        model, 
        optimizer, 
        train_loader, 
        train_loader,  # Use same loader for source and target
        args.device, 
        args.epochs,
        args.output_dir
    )
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.2f} minutes")
    
    # Plot and save loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(loss_curve, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'loss_curve.png'))
    plt.close()
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'final_model.pt')
    try:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': vars(args),
            'loss_curve': loss_curve
        }, final_model_path)
        print(f"Saved final model to {final_model_path}")
    except Exception as e:
        print(f"Error saving final model: {str(e)}")
        traceback.print_exc()
    
    # Optionally apply reflow procedure for better one-step generation
    if args.reflow_steps > 0:
        print(f"Applying reflow procedure with {args.reflow_steps} steps")
        
        # Get a batch of source images for reflow
        source_batch, _ = next(iter(train_loader))
        source_batch = source_batch.to(args.device)
        
        # Apply reflow
        rf.reflow(
            source_batch,
            training_steps=args.reflow_steps,
            batch_size=min(16, source_batch.size(0)),
            lr=args.learning_rate * 0.1,
            device=args.device
        )
        
        # Save reflowed model
        reflowed_model_path = os.path.join(args.output_dir, 'reflowed_model.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': vars(args),
            'reflow_steps': args.reflow_steps
        }, reflowed_model_path)
        
        print(f"Reflowed model saved to {reflowed_model_path}")
        
        # Visualize results after reflow
        visualize_results(model, rf, val_loader, args.epochs + 1, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MONAI-based Rectified Flow for MRI Translation')
    
    # Data arguments
    parser.add_argument('--t1_dir', type=str, required=True, help='Directory containing T1 slices')
    parser.add_argument('--t2_dir', type=str, required=True, help='Directory containing T2 slices')
    
    # Model arguments
    parser.add_argument('--features', type=int, nargs='+', default=[32, 64, 128, 256],
                        help='Features for UNet layers')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda or cpu)')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
    
    # Rectified Flow arguments
    parser.add_argument('--num_steps', type=int, default=50, help='Number of steps for ODE solution')
    parser.add_argument('--reflow_steps', type=int, default=100, 
                       help='Number of reflow steps (0 to disable)')
    
    args = parser.parse_args()
    main(args)