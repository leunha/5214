import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18  # Lighter backbone than DenseNet121
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

class EfficientPretrainedFlow(nn.Module):
    """
    Efficient Rectified Flow model for T1-to-T2 MRI translation using a pre-trained ResNet18
    backbone with 95% frozen parameters for extremely fast training.
    """
    def __init__(self, img_size=256, in_channels=1, out_channels=1, freeze_ratio=0.95, pretrained=True):
        """
        Initialize the EfficientPretrainedFlow model.
        
        Args:
            img_size: Size of input images (assumed square)
            in_channels: Number of input channels (1 for grayscale MRI)
            out_channels: Number of output channels (1 for velocity field)
            freeze_ratio: Proportion of backbone layers to freeze (0.0-1.0)
            pretrained: Whether to use pre-trained weights
        """
        super().__init__()
        self.img_size = img_size
        
        # 1. Load pre-trained ResNet18 backbone (much lighter than DenseNet121)
        resnet = resnet18(pretrained=pretrained)
        
        # 2. Modify first conv layer to accept grayscale input + time channel
        original_conv1 = resnet.conv1
        resnet.conv1 = nn.Conv2d(
            in_channels + 1,  # Single channel MRI + time channel
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        
        # If using pre-trained weights, initialize the modified conv layer
        if pretrained:
            with torch.no_grad():
                # Average the RGB weights for the grayscale channel
                new_weight = resnet.conv1.weight.clone()
                new_weight[:, 0:1, :, :] = original_conv1.weight.mean(dim=1, keepdim=True)
                # Initialize time channel with small random values
                new_weight[:, 1:2, :, :] = torch.randn_like(new_weight[:, 1:2, :, :]) * 0.01
                resnet.conv1.weight.copy_(new_weight)
        
        # 3. Create encoder from ResNet (without FC layer)
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # output: [B, 64, H/4, W/4]
            resnet.layer2,  # output: [B, 128, H/8, W/8]
            resnet.layer3,  # output: [B, 256, H/16, W/16]
            resnet.layer4,  # output: [B, 512, H/32, W/32]
        )
        
        # 4. Freeze encoder layers based on freeze_ratio
        self._freeze_layers(freeze_ratio)
        
        # 5. Simplified decoder pathway (optimized for speed)
        self.decoder = nn.Sequential(
            # Upsample from 512 -> 256
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Upsample from 256 -> 128
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Upsample from 128 -> 64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Upsample from 64 -> 32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Upsample from 32 -> 16
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # Final 1x1 conv to get desired output channels
            nn.Conv2d(16, out_channels, kernel_size=1),
        )
        
        # 6. Simple time embedding (minimal parameters)
        self.time_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
        )
        
    def _freeze_layers(self, freeze_ratio):
        """
        Freeze encoder layers based on freeze_ratio.
        
        Args:
            freeze_ratio: Proportion of layers to freeze (0.0-1.0)
        """
        # Calculate total parameters
        total_params = sum(p.numel() for p in self.parameters())
        
        # Count encoder parameters
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        
        # Calculate how many encoder parameters to freeze
        freeze_params = int(total_params * freeze_ratio)
        
        # Start freezing from earliest layers
        frozen_params = 0
        
        # Freeze entire encoder if it's less than our target
        if encoder_params <= freeze_params:
            for param in self.encoder.parameters():
                param.requires_grad = False
            frozen_params = encoder_params
            print(f"Frozen entire encoder: {frozen_params:,} params")
            return
            
        # Otherwise, freeze layer by layer
        for name, module in self.encoder.named_children():
            layer_params = sum(p.numel() for p in module.parameters())
            
            # Check if freezing this layer would exceed our target
            if frozen_params + layer_params > freeze_params:
                # Freeze a portion of this layer
                params_to_freeze = freeze_params - frozen_params
                frozen_count = 0
                
                for param_name, param in module.named_parameters():
                    param_size = param.numel()
                    if frozen_count + param_size <= params_to_freeze:
                        param.requires_grad = False
                        frozen_count += param_size
                
                frozen_params += frozen_count
                print(f"Partially frozen layer {name}: {frozen_count:,} params")
                break
            else:
                # Freeze the entire layer
                for param in module.parameters():
                    param.requires_grad = False
                frozen_params += layer_params
                print(f"Frozen layer {name}: {layer_params:,} params")
        
        # Report freezing statistics
        trainable_params = total_params - frozen_params
        print(f"Frozen {frozen_params:,}/{total_params:,} parameters ({frozen_params/total_params*100:.1f}%)")
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    def forward(self, x, t):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor [B, C, H, W]
            t: Time step [B]
            
        Returns:
            Velocity field prediction [B, C, H, W]
        """
        # 1. Reshape time to match x dimensions and concatenate
        time_channel = t.view(-1, 1, 1, 1).expand(-1, 1, x.shape[2], x.shape[3])
        x_t = torch.cat([x, time_channel], dim=1)
        
        # 2. Encode
        features = self.encoder(x_t)
        
        # 3. Decode
        output = self.decoder(features)
        
        # Resize output to match input if needed
        if output.shape[2:] != x.shape[2:]:
            output = F.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=False)
            
        return output

class EfficientRectifiedFlowODE:
    """
    Efficient ODE solver for the Rectified Flow model.
    """
    def __init__(self, model, num_steps=50):
        self.model = model
        self.num_steps = num_steps
    
    def sample_ode(self, z0, N=None):
        """
        Solve the ODE from initial state z0.
        
        Args:
            z0: Initial state (B, C, H, W)
            N: Number of steps (if None, use self.num_steps)
            
        Returns:
            list: Trajectory of states [z0, z1, ..., zN]
        """
        if N is None:
            N = self.num_steps
        
        device = z0.device
        trajectories = [z0]
        z = z0
        
        dt = 1.0 / N
        for i in range(N):
            t = torch.ones(z.shape[0], device=device) * (i * dt)
            with torch.no_grad():  # No grad needed for sampling
                dz = self.model(z, t)
            z = z + dz * dt
            trajectories.append(z.clone())
        
        return trajectories

def create_optimizer(model, base_lr=1e-4):
    """
    Create an optimizer with different learning rates for pre-trained and new parameters.
    
    Args:
        model: EfficientPretrainedFlow model
        base_lr: Base learning rate for new parameters
        
    Returns:
        optimizer: Optimizer with parameter groups
    """
    # Separate parameters: pre-trained backbone vs. new decoder
    pretrained_params = []
    new_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'encoder' in name:
                pretrained_params.append(param)
            else:
                new_params.append(param)
    
    # Create optimizer with different learning rates
    optimizer = torch.optim.AdamW([
        {'params': pretrained_params, 'lr': base_lr * 0.05},  # Much lower LR for pre-trained
        {'params': new_params, 'lr': base_lr}                # Normal LR for new params
    ])
    
    return optimizer

def quick_train(model, train_loader, val_loader, device, output_dir, 
               num_epochs=1, save_every=1, early_stop_patience=None):
    """
    Train the model for a short time period.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        output_dir: Directory to save outputs
        num_epochs: Number of epochs to train
        save_every: Save checkpoint every N epochs
        early_stop_patience: Stop training if validation loss doesn't improve for N epochs
        
    Returns:
        model: Trained model
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    
    # Create optimizer with appropriate learning rates
    optimizer = create_optimizer(model, base_lr=1e-3)
    
    # Optional learning rate scheduler for quick adaptation
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1, verbose=True
    )
    
    # Initialize ODE solver for visualization
    ode_solver = EfficientRectifiedFlowODE(model, num_steps=20)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Get a fixed batch for visualization
    val_batch_x, val_batch_y = next(iter(val_loader))
    val_batch_x, val_batch_y = val_batch_x.to(device), val_batch_y.to(device)
    
    print(f"Starting quick training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        # Progress bar for training
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_data in progress:
            # Get source and target
            source_batch, target_batch = batch_data
            source_batch, target_batch = source_batch.to(device), target_batch.to(device)
            
            # Sample random time steps
            t = torch.rand(source_batch.size(0), device=device)
            
            # Interpolate between source and target
            z_t = source_batch * (1-t.view(-1,1,1,1)) + target_batch * t.view(-1,1,1,1)
            
            # Forward pass
            optimizer.zero_grad()
            pred = model(z_t, t)
            
            # Compute loss
            loss = F.mse_loss(pred, target_batch - source_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update progress
            epoch_loss += loss.item()
            batch_count += 1
            progress.set_postfix({"loss": loss.item()})
        
        avg_train_loss = epoch_loss / batch_count
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_batch_count = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                source_batch, target_batch = batch_data
                source_batch, target_batch = source_batch.to(device), target_batch.to(device)
                
                # Sample random time steps
                t = torch.rand(source_batch.size(0), device=device)
                
                # Interpolate
                z_t = source_batch * (1-t.view(-1,1,1,1)) + target_batch * t.view(-1,1,1,1)
                
                # Forward pass
                pred = model(z_t, t)
                loss = F.mse_loss(pred, target_batch - source_batch)
                
                val_loss += loss.item()
                val_batch_count += 1
        
        avg_val_loss = val_loss / val_batch_count
        val_losses.append(avg_val_loss)
        
        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Generate and save sample visualizations
        print(f"Generating visualizations for epoch {epoch+1}...")
        vis_path = os.path.join(output_dir, 'visualizations', f'epoch_{epoch+1}.png')
        generate_visualization(model, ode_solver, val_batch_x, val_batch_y, vis_path)
        print(f"Saved visualization to {vis_path}")
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pt')
            try:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
            except Exception as e:
                print(f"Error saving checkpoint: {str(e)}")
        
        # Check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            best_model_path = os.path.join(output_dir, 'best_model.pt')
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model to {best_model_path}")
        else:
            patience_counter += 1
        
        if early_stop_patience is not None and patience_counter >= early_stop_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Save final model
    final_model_path = os.path.join(output_dir, 'final_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, final_model_path)
    print(f"Saved final model to {final_model_path}")
    
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'))
    plt.close()
    print(f"Saved loss curves to {os.path.join(output_dir, 'loss_curves.png')}")
    
    print(f"Training completed. Final model saved to {final_model_path}")
    return model

def generate_visualization(model, ode_solver, source_batch, target_batch, output_path):
    """
    Generate visualization of the model's predictions.
    
    Args:
        model: Trained model
        ode_solver: ODE solver
        source_batch: Batch of source images
        target_batch: Batch of target images
        output_path: Path to save visualization
    """
    print(f"Starting visualization generation, saving to {output_path}")
    model.eval()
    
    try:
        with torch.no_grad():
            # Generate trajectory
            print("Generating trajectories...")
            trajectories = ode_solver.sample_ode(source_batch[:4])
            generated = trajectories[-1]
            print(f"Generated {len(trajectories)} trajectory steps")
            
            # Create figure
            n_samples = min(4, source_batch.size(0))
            fig, axes = plt.subplots(3, n_samples, figsize=(3*n_samples, 9))
            
            # For each sample
            for i in range(n_samples):
                # Show source (T1)
                axes[0, i].imshow(source_batch[i, 0].cpu().numpy(), cmap='gray')
                axes[0, i].set_title('T1 Source')
                axes[0, i].axis('off')
                
                # Show generated (predicted T2)
                axes[1, i].imshow(generated[i, 0].cpu().numpy(), cmap='gray')
                axes[1, i].set_title('Generated T2')
                axes[1, i].axis('off')
                
                # Show target (true T2)
                axes[2, i].imshow(target_batch[i, 0].cpu().numpy(), cmap='gray')
                axes[2, i].set_title('True T2')
                axes[2, i].axis('off')
            
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            print(f"Created output directory {os.path.dirname(output_path)}")
            
            # Add row labels
            if n_samples > 0:
                axes[0, 0].set_ylabel("Source (T1)", fontsize=12)
                axes[1, 0].set_ylabel("Generated (T2)", fontsize=12)
                axes[2, 0].set_ylabel("True T2", fontsize=12)
            
            plt.tight_layout()
            print(f"Saving figure to {output_path}...")
            plt.savefig(output_path)
            print(f"Figure saved successfully to {output_path}")
            plt.close()
            
            # Also save a trajectory visualization
            if len(trajectories) > 2:
                traj_path = output_path.replace('.png', '_trajectory.png')
                print(f"Creating trajectory visualization at {traj_path}")
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
                plt.savefig(traj_path)
                print(f"Trajectory visualization saved to {traj_path}")
                plt.close()
                
        print("Visualization generation completed successfully!")
    except Exception as e:
        import traceback
        print(f"Error generating visualization: {str(e)}")
        traceback.print_exc()

def main():
    from dataset import create_data_loaders
    import argparse
    import os  # Ensure os is imported here too
    
    parser = argparse.ArgumentParser(description='Quick train pretrained Rectified Flow for MRI Translation')
    
    # Data arguments
    parser.add_argument('--t1_dir', type=str, required=True, help='Directory containing T1 slices')
    parser.add_argument('--t2_dir', type=str, required=True, help='Directory containing T2 slices')
    
    # Model arguments
    parser.add_argument('--freeze_ratio', type=float, default=0.95, 
                        help='Proportion of parameters to freeze (0.0-1.0)')
    parser.add_argument('--img_size', type=int, default=256, help='Image size')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Base learning rate')
    parser.add_argument('--num_workers', type=int, default=0, 
                        help='Number of workers for data loading (0 for CPU training)')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cuda or cpu)')
    parser.add_argument('--output_dir', type=str, default='./quick_results', 
                        help='Directory to save results')
    parser.add_argument('--early_stop', type=int, default=None, 
                        help='Stop training if validation loss doesnt improve for N epochs')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create output directories
    print(f"Creating output directory: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    print(f"Created visualization directory: {vis_dir}")
    
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
    
    # Create model
    model = EfficientPretrainedFlow(
        img_size=args.img_size,
        in_channels=1,
        out_channels=1,
        freeze_ratio=args.freeze_ratio,
        pretrained=True
    ).to(device)
    
    # Train model
    model = quick_train(
        model,
        train_loader,
        val_loader,
        device,
        args.output_dir,
        num_epochs=args.epochs,
        early_stop_patience=args.early_stop
    )
    
    print("Quick training completed!")

if __name__ == "__main__":
    main() 