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
import logging
import pandas as pd

from dataset import MRISliceDataset, create_data_loaders
from monai_rectified_flow import MonaiRectifiedFlow, RectifiedFlowODE, SSIM

from evaluate_monai import evaluate_model

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
    logging.info(f"Checkpoint saved to {checkpoint_path}")

def visualize_results(model, rf, data_loader, epoch, args):
    """Visualize sample results during training"""
    model.eval()
    
    # Handle args as either dictionary or object with attributes
    output_dir = args['output_dir'] if isinstance(args, dict) else args.output_dir
    device = args['device'] if isinstance(args, dict) else args.device
    num_steps = args['num_steps'] if isinstance(args, dict) else args.num_steps
    
    # Create visualization directory
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    
    with torch.no_grad():
        # Get one batch from validation set
        source_batch, target_batch = next(iter(data_loader))
        source_batch, target_batch = source_batch.to(device), target_batch.to(device)
        
        # Generate trajectory
        trajectories = rf.sample_ode(source_batch, N=num_steps)
        source = source_batch
        target = target_batch
        generated = trajectories[-1]  # Final state in the trajectory
        
        # Create figure for side-by-side comparison
        n_samples = min(4, source.size(0))
        fig, axes = plt.subplots(4, n_samples, figsize=(4*n_samples, 12))
        
        # For each sample
        for i in range(n_samples):
            # Handle the case where n_samples is 1
            if n_samples == 1:
                ax_row = axes
            else:
                ax_row = axes[:, i]
                
            # Show source (T1)
            ax_row[0].imshow(source[i, 0].cpu().numpy(), cmap='gray')
            ax_row[0].set_title('Source (T1)')
            ax_row[0].axis('off')
            
            # Show generated (predicted T2)
            ax_row[1].imshow(generated[i, 0].cpu().numpy(), cmap='gray')
            ax_row[1].set_title('Generated (T2)')
            ax_row[1].axis('off')
            
            # Show target (true T2)
            ax_row[2].imshow(target[i, 0].cpu().numpy(), cmap='gray')
            ax_row[2].set_title('Target (True T2)')
            ax_row[2].axis('off')
            
            # Show overlay of target and generated for alignment check
            ax_row[3].imshow(target[i, 0].cpu().numpy(), cmap='gray')
            ax_row[3].imshow(generated[i, 0].cpu().numpy(), cmap='hot', alpha=0.5)
            ax_row[3].set_title('True/Generated Overlay')
            ax_row[3].axis('off')
        
        # Add row labels
        if n_samples > 0:
            if n_samples == 1:
                axes[0].set_ylabel("Source (T1)", fontsize=12)
                axes[1].set_ylabel("Generated (T2)", fontsize=12)
                axes[2].set_ylabel("Target (T2)", fontsize=12)
                axes[3].set_ylabel("Alignment Check", fontsize=12)
            else:
                axes[0, 0].set_ylabel("Source (T1)", fontsize=12)
                axes[1, 0].set_ylabel("Generated (T2)", fontsize=12)
                axes[2, 0].set_ylabel("Target (T2)", fontsize=12)
                axes[3, 0].set_ylabel("Alignment Check", fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'visualizations', f'results_epoch_{epoch}.png'))
        plt.close()
        
        # Visualize trajectory evolution
        # Select a subset of steps from the trajectory
        step_indices = np.linspace(0, len(trajectories) - 1, min(8, len(trajectories))).astype(int)
        
        # Create figure
        fig, axes = plt.subplots(1, len(step_indices), figsize=(20, 4))
        
        for i, idx in enumerate(step_indices):
            # Handle the case of single subplot or multiple subplots properly
            ax = axes if len(step_indices) == 1 else axes[i]
                
            ax.imshow(trajectories[idx][0, 0].cpu().numpy(), cmap='gray')
            ax.set_title(f'Step {idx}')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'visualizations', f'trajectory_epoch_{epoch}.png'))
        plt.close()

def compute_validation_metrics(rf, val_loader, device, use_combined_loss=False, ssim_calculator=None):
    """Compute validation metrics"""
    rf.model.eval()
    val_losses = []
    metrics = {'l1': [], 'ssim': [], 'rf': []}
    
    if ssim_calculator is None and use_combined_loss:
        ssim_calculator = SSIM().to(device)
    
    with torch.no_grad():
        for val_data in val_loader:
            source_val, target_val = val_data
            source_val, target_val = source_val.to(device), target_val.to(device)
            
            # Sample random timesteps
            t = torch.rand(source_val.size(0), device=device)
            
            # Interpolate between source and target
            z_t = source_val * (1-t.view(-1,1,1,1)) + target_val * t.view(-1,1,1,1)
            
            # Forward pass
            pred = rf.model(z_t, t)
            
            # Basic RF loss
            rf_loss = F.mse_loss(pred, target_val - source_val)
            
            # Predicted T2
            pred_t2 = source_val + pred
            
            # L1 metric
            l1_error = F.l1_loss(pred_t2, target_val).item()
            metrics['l1'].append(l1_error)
            
            # SSIM metric if required
            if use_combined_loss and ssim_calculator is not None:
                ssim_val = ssim_calculator(pred_t2, target_val).item()
                metrics['ssim'].append(ssim_val)
            
            metrics['rf'].append(rf_loss.item())
            val_losses.append(rf_loss.item())
    
    # Calculate average validation loss
    avg_val_loss = sum(val_losses) / len(val_losses)
    
    # Calculate average metrics
    avg_metrics = {}
    for key, values in metrics.items():
        if values:
            avg_metrics[key] = sum(values) / len(values)
    
    return avg_val_loss, avg_metrics

def train_with_validation(rf, optimizer, train_loader, val_loader, device, epochs, 
                        scheduler=None, output_dir=None, use_combined_loss=True,
                        rf_weight=1.0, l1_weight=0.5, ssim_weight=0.5,
                        early_stopping_patience=5):
    """
    Train the Rectified Flow model with validation after each epoch.
    
    Args:
        rf: RectifiedFlowODE model
        optimizer: Optimizer
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        device: Device for training
        epochs: Number of epochs
        scheduler: Learning rate scheduler
        output_dir: Directory to save checkpoints
        use_combined_loss: Whether to use the combined loss function
        rf_weight: Weight for Rectified Flow loss
        l1_weight: Weight for L1 loss
        ssim_weight: Weight for SSIM loss
        early_stopping_patience: Number of epochs to wait before early stopping
        
    Returns:
        model: Trained model
        metrics: Dictionary of training and validation metrics
    """
    rf.model.train()
    
    # Set up metrics tracking
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'best_val_loss': float('inf'),
        'epochs_without_improvement': 0
    }
    
    # Create SSIM calculator for the combined loss
    ssim_calculator = SSIM().to(device)
    
    for epoch in range(epochs):
        # Training phase
        rf.model.train()
        epoch_losses = []
        epoch_loss_breakdown = {'rf': 0.0, 'l1': 0.0, 'ssim': 0.0, 'ssim_value': 0.0}
        num_batches = 0
        
        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
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
            pred = rf.model(z_t, t)
            
            # Compute loss with standard or combined approach
            if use_combined_loss:
                # 1. Rectified Flow Loss (velocity matching)
                target_velocity = target_batch - source_batch
                rf_loss = F.mse_loss(pred, target_velocity)
                
                # 2. Predicted T2 estimation
                pred_t2 = source_batch + pred
                
                # 3. L1 Loss (pixel-level difference)
                l1_loss = F.l1_loss(pred_t2, target_batch)
                
                # 4. SSIM Loss (structural similarity)
                ssim_value = ssim_calculator(pred_t2, target_batch)
                ssim_loss = 1.0 - ssim_value
                
                # 5. Combine losses with weights
                loss = rf_weight * rf_loss + l1_weight * l1_loss + ssim_weight * ssim_loss
                
                # 6. Update loss breakdown
                epoch_loss_breakdown['rf'] += rf_loss.item()
                epoch_loss_breakdown['l1'] += l1_loss.item()
                epoch_loss_breakdown['ssim'] += ssim_loss.item()
                epoch_loss_breakdown['ssim_value'] += ssim_value.item()
            else:
                # Original MSE loss
                loss = F.mse_loss(pred, target_batch - source_batch)
            
            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            num_batches += 1
        
        # Calculate average loss for the epoch
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        metrics['train_loss'].append(avg_loss)
        
        # Validation phase
        val_loss, val_metrics = compute_validation_metrics(
            rf, val_loader, device, use_combined_loss, ssim_calculator
        )
        metrics['val_loss'].append(val_loss)
        
        # Update learning rate scheduler if provided
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Print loss breakdown
        logging.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        if use_combined_loss and num_batches > 0:
            logging.info(f"  Train RF Loss: {epoch_loss_breakdown['rf']/num_batches:.6f}, " +
                  f"L1 Loss: {epoch_loss_breakdown['l1']/num_batches:.6f}, " +
                  f"SSIM Loss: {epoch_loss_breakdown['ssim']/num_batches:.6f}, " +
                  f"SSIM Value: {1-epoch_loss_breakdown['ssim_value']/num_batches:.4f}")
            
            if 'ssim' in val_metrics:
                logging.info(f"  Val SSIM: {val_metrics['ssim']:.4f}, Val L1: {val_metrics['l1']:.6f}")
        
        # Save model if validation improves
        if val_loss < metrics['best_val_loss']:
            metrics['best_val_loss'] = val_loss
            metrics['epochs_without_improvement'] = 0
            
            # Save best model
            if output_dir is not None:
                best_model_path = os.path.join(output_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': rf.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, best_model_path)
                logging.info(f"Saved best model to {best_model_path}")
                
            # Visualize with the best model
            visualize_results(rf.model, rf, val_loader, epoch + 1, {
                'device': device, 
                'output_dir': output_dir,    
                'num_steps': getattr(rf, 'num_steps', 20)  # Default to 20 if not set
            })
        else:
            metrics['epochs_without_improvement'] += 1
            logging.info(f"No improvement for {metrics['epochs_without_improvement']} epochs.")
        
        # Save regular checkpoint
        if output_dir is not None:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pt')
            try:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': rf.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'val_loss': val_loss,
                }, checkpoint_path)
                logging.info(f"Saved checkpoint to {checkpoint_path}")
            except Exception as e:
                logging.error(f"Error saving checkpoint: {str(e)}")
        
        # Early stopping check
        if metrics['epochs_without_improvement'] >= early_stopping_patience:
            logging.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    return rf, metrics

def main(args):
    # Set up logging
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.output_dir, 'training.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Add console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    # Set device
    if args.device is None:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)
    logging.info(f"Using device: {args.device}")
    
    # Set up output directory with absolute path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.abspath(os.path.join(args.output_dir, f"run_{timestamp}"))
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Output directory: {args.output_dir}")
    
    # Create visualizations directory
    vis_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    logging.info(f"Visualization directory: {vis_dir}")
    
    # Save configuration
    config_path = os.path.join(args.output_dir, 'config.txt')
    with open(config_path, 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    logging.info(f"Saved configuration to {config_path}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        args.t1_dir,
        args.t2_dir,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        num_workers=args.num_workers
    )
    
    logging.info(f"Training on {len(train_loader.dataset)} samples")
    logging.info(f"Validating on {len(val_loader.dataset)} samples")
    
    # Check batch data shapes
    sample_batch = next(iter(train_loader))
    source_batch, target_batch = sample_batch
    logging.info(f"Source shape: {source_batch.shape}, Target shape: {target_batch.shape}")
    
    # Create model
    model = MonaiRectifiedFlow(
        img_size=source_batch.shape[2],
        in_channels=1,
        out_channels=1,
        features=args.features
    ).to(args.device)

    rf = RectifiedFlowODE(model, num_steps=args.num_steps)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model created with {total_params:,} parameters ({trainable_params:,} trainable)")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Create learning rate scheduler if requested
    scheduler = None
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        logging.info("Using ReduceLROnPlateau learning rate scheduler")
    
    # Train the model with validation
    start_time = time.time()
    
    rf, metrics = train_with_validation( 
        rf, 
        optimizer, 
        train_loader, 
        val_loader, 
        args.device, 
        args.epochs,
        scheduler=scheduler,
        output_dir=args.output_dir, 
        use_combined_loss=args.use_combined_loss,
        rf_weight=args.rf_weight,
        l1_weight=args.l1_weight,
        ssim_weight=args.ssim_weight,
        early_stopping_patience=args.early_stopping_patience
    )
    
    total_time = time.time() - start_time
    logging.info(f"Training completed in {total_time/60:.2f} minutes")
    
    # Plot and save loss curve with train/val
    plt.figure(figsize=(10, 5))
    plt.plot(metrics['train_loss'], label='Training Loss')
    plt.plot(metrics['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'loss_curve.png'))
    plt.close()
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'final_model.pt')
    try:
        torch.save({
            'model_state_dict': rf.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': vars(args),
            'metrics': metrics,
        }, final_model_path)
        logging.info(f"Saved final model to {final_model_path}")
    except Exception as e:
        logging.error(f"Error saving final model: {str(e)}")
        import traceback
        traceback.print_exc()

    evaluate_model(
        rf.model,
        test_loader,
        args.device,
        num_steps=args.num_steps,
        output_dir=args.output_dir,
        reflow=0
    )
    evaluate_model(
        rf.model,
        test_loader,
        args.device,
        num_steps=1,
        output_dir=args.output_dir,
        reflow=0
    )
    
    # Optionally apply reflow procedure for better one-step generation
    if args.reflow_steps > 0:
        for i in range(1, args.reflow_steps+1):
            logging.info(f"Applying reflow procedure with {i} steps")
            
            # Get a batch of source images for reflow
            source_batch, _ = next(iter(train_loader))
            source_batch = source_batch.to(args.device)
            
            # Apply reflow
            rf.reflow(
                source_batch,
                training_steps=1,
                batch_size=min(16, source_batch.size(0)),
                lr=args.learning_rate * 0.1,
                device=args.device
            )
            
            # Save reflowed model
            reflowed_model_path = os.path.join(args.output_dir, f'reflowed_model_{i}.pt')
            torch.save({
                'model_state_dict': rf.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': vars(args),
                'reflow_steps': i
            }, reflowed_model_path)
            
            logging.info(f"Reflowed model saved to {reflowed_model_path}")

            evaluate_model(
                rf.model,
                test_loader,
                args.device,
                num_steps=args.num_steps,
                output_dir=args.output_dir,
                reflow=i
            )
            evaluate_model(
                rf.model,
                test_loader,
                args.device,
                num_steps=1,
                output_dir=args.output_dir,
                reflow=i
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MONAI-based Rectified Flow for MRI Translation')
    
    # Data arguments
    parser.add_argument('--t1_dir', type=str, required=True, help='Directory containing T1 slices')
    parser.add_argument('--t2_dir', type=str, required=True, help='Directory containing T2 slices')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Training ratio for train/val split')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation ratio for train/val split')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Test ratio for train/val split')


    # Model arguments
    parser.add_argument('--features', type=int, nargs='+', default=[32, 64, 128, 256],
                        help='Features for UNet layers')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=2, 
                       help='Batch size for training (use 1-2 for CPU)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=2, 
                       help='Number of workers for data loading (use 0-1 for CPU)')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda or cpu)')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--use_scheduler', action='store_true', 
                       help='Use learning rate scheduler')
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                       help='Number of epochs with no improvement before early stopping')
    
    # Rectified Flow arguments
    parser.add_argument('--num_steps', type=int, default=20, 
                       help='Number of steps for ODE solution (10-20 good for CPU)')
    parser.add_argument('--reflow_steps', type=int, default=1, 
                       help='Number of reflow steps (0 to disable, 1-3 recommended)')
    
    # Loss function arguments
    parser.add_argument('--use_combined_loss', action='store_true', 
                       help='Use combined loss (RF + L1 + SSIM)')
    parser.add_argument('--rf_weight', type=float, default=1.0, 
                       help='Weight for Rectified Flow loss')
    parser.add_argument('--l1_weight', type=float, default=0.5, 
                       help='Weight for L1 loss')
    parser.add_argument('--ssim_weight', type=float, default=0.5, 
                       help='Weight for SSIM loss')
    
    args = parser.parse_args()
    main(args)