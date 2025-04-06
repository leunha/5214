import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from tqdm import tqdm
import numpy as np
import os

# Add SSIM implementation
class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images"""
    def __init__(self, win_size=11, k1=0.01, k2=0.03):
        """
        Args:
            win_size: Window size for SSIM calculation
            k1, k2: SSIM parameters
        """
        super(SSIM, self).__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer('w', self._create_window(win_size))
        self.cov_norm = win_size * win_size

    def _create_window(self, win_size):
        # Create a 2D Gaussian window
        gauss = torch.Tensor([np.exp(-(x - win_size//2)**2/float(win_size)) for x in range(win_size)])
        gauss = gauss / gauss.sum()
        _1D_window = gauss.unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        return _2D_window
    
    def forward(self, x, y):
        """
        Calculate SSIM between x and y
        Args:
            x, y: Images to compare (B, C, H, W)
        Returns:
            SSIM value (higher is better, range: [0,1])
        """
        # Check size
        assert x.shape == y.shape, f"Input images must have the same dimensions: {x.shape} vs {y.shape}"
        
        # Add batch and channel dimensions if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(0).unsqueeze(0)
            y = y.unsqueeze(0).unsqueeze(0)
        elif len(x.shape) == 3:
            x = x.unsqueeze(1)
            y = y.unsqueeze(1)
            
        # Calculate means
        w = self.w.expand(x.shape[1], -1, -1, -1)
        
        # Compute means
        mu_x = F.conv2d(x, w, padding=self.win_size//2, groups=x.shape[1])
        mu_y = F.conv2d(y, w, padding=self.win_size//2, groups=y.shape[1])
        
        # Compute variances and covariance
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y
        
        sigma_x_sq = F.conv2d(x * x, w, padding=self.win_size//2, groups=x.shape[1]) - mu_x_sq
        sigma_y_sq = F.conv2d(y * y, w, padding=self.win_size//2, groups=y.shape[1]) - mu_y_sq
        sigma_xy = F.conv2d(x * y, w, padding=self.win_size//2, groups=x.shape[1]) - mu_xy
        
        # SSIM constants
        C1 = (self.k1 * 1.0) ** 2
        C2 = (self.k2 * 1.0) ** 2
        
        # Calculate SSIM
        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
                  ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
        
        # Return mean SSIM
        return ssim_map.mean()

class MonaiRectifiedFlow(nn.Module):
    """
    Rectified Flow model for MRI image-to-image translation
    using MONAI's UNet as the backbone.
    """
    def __init__(self, img_size=256, in_channels=1, out_channels=1, features=(32, 64, 128, 256)):
        super().__init__()
        self.img_size = img_size
        
        # MONAI UNet as backbone
        self.unet = UNet(
            spatial_dims=2,  # 2D UNet
            in_channels=in_channels + 1,  # +1 for time channel
            out_channels=out_channels,
            channels=features,
            strides=(2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH
        )
    
    def forward(self, x, t):
        # Reshape time to appropriate format and expand to spatial dimensions
        t = t.view(-1, 1, 1, 1).expand(-1, 1, self.img_size, self.img_size)
        
        # Concatenate input image and time along channel dimension
        x_t = torch.cat([x, t], dim=1)
        
        # Forward pass through UNet
        return self.unet(x_t)

class RectifiedFlowODE:
    """
    Rectified Flow ODE solver for MRI translation.
    """
    def __init__(self, model, num_steps=100):
        """
        Initialize the Rectified Flow ODE.
        
        Args:
            model: Neural network model to predict velocity field
            num_steps: Number of steps for ODE integration
        """
        self.model = model
        self.num_steps = num_steps
        self.ssim = SSIM()
    
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
            dz = self.model(z, t)
            z = z + dz * dt
            trajectories.append(z.clone())
        
        return trajectories
    
    def reflow(self, z0_samples, training_steps, batch_size, lr=1e-4, device='cuda'):
        """
        Train with reflow procedure.
        
        Args:
            z0_samples: Initial samples
            training_steps: Number of training steps
            batch_size: Batch size
            lr: Learning rate
            device: Device for training
            
        Returns:
            self: For chaining
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        for step in tqdm(range(training_steps), desc="Reflow Training"):
            # Sample random batch
            idx = torch.randperm(len(z0_samples))[:batch_size]
            z0_batch = z0_samples[idx]
            
            # Forward pass
            trajectories = self.sample_ode(z0_batch)
            z1 = trajectories[-1]
            
            # Backward pass
            reverse_trajectories = self.sample_ode(z1)
            z0_recon = reverse_trajectories[-1]
            
            # Compute loss
            loss = F.mse_loss(z0_recon, z0_batch)
            
            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % 10 == 0:
                print(f"Reflow Step {step}/{training_steps}, Loss: {loss.item():.6f}")
        
        return self

def calculate_combined_loss(pred_velocity, source, target, t, ssim_calculator=None, rf_weight=1.0, l1_weight=0.5, ssim_weight=0.5):
    """
    Calculate combined loss for Rectified Flow MRI translation
    
    Args:
        pred_velocity: Predicted velocity field
        source: Source image (T1)
        target: Target image (T2)
        t: Current timestep
        ssim_calculator: SSIM loss calculator
        rf_weight: Weight for Rectified Flow loss
        l1_weight: Weight for L1 loss 
        ssim_weight: Weight for SSIM loss
        
    Returns:
        total_loss: Combined weighted loss
        loss_dict: Dictionary of individual losses
    """
    # Create SSIM calculator if not provided
    if ssim_calculator is None:
        ssim_calculator = SSIM().to(pred_velocity.device)
    
    # 1. Rectified Flow Loss (velocity matching)
    target_velocity = target - source  # Straight-line path target
    rf_loss = F.mse_loss(pred_velocity, target_velocity)
    
    # 2. Predicted T2 estimation using the predicted velocity
    # This is a simplification; during training we interpolate but here we approximate
    pred_t2 = source + pred_velocity  # Simple addition as approximation
    
    # 3. L1 Loss (pixel-level difference)
    l1_loss = F.l1_loss(pred_t2, target)
    
    # 4. SSIM Loss (structural similarity)
    ssim_value = ssim_calculator(pred_t2, target)
    ssim_loss = 1.0 - ssim_value  # Convert to loss (1 - SSIM)
    
    # Combine losses with weights
    total_loss = rf_weight * rf_loss + l1_weight * l1_loss + ssim_weight * ssim_loss
    
    # Return loss breakdown for logging
    loss_dict = {
        'total': total_loss.item(),
        'rf': rf_loss.item(),
        'l1': l1_loss.item(),
        'ssim': ssim_loss.item(),
        'ssim_value': ssim_value.item()
    }
    
    return total_loss, loss_dict

def train_monai_rectified_flow(rf, optimizer, source_loader, target_loader, device, epochs, output_dir=None, use_combined_loss=True):
    """
    Train the Rectified Flow model.
    
    Args:
        rf: RectifiedFlowODE model
        optimizer: Optimizer
        source_loader: DataLoader for source domain (T1)
        target_loader: DataLoader for target domain (T2)
        device: Device for training
        epochs: Number of epochs
        output_dir: Directory to save checkpoints
        use_combined_loss: Whether to use the combined loss function
        
    Returns:
        model: Trained model
        loss_curve: Training loss history
    """
    rf.model.train()
    loss_curve = []
    
    # Create SSIM calculator for the combined loss
    ssim_calculator = SSIM().to(device)
    
    for epoch in range(epochs):
        epoch_losses = []
        epoch_loss_breakdown = {'rf': 0.0, 'l1': 0.0, 'ssim': 0.0, 'ssim_value': 0.0}
        num_batches = 0
        
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
            pred = rf.model(z_t, t)
            
            # Compute loss with standard or combined approach
            if use_combined_loss:
                loss, loss_dict = calculate_combined_loss(
                    pred, source_batch, target_batch, t, 
                    ssim_calculator=ssim_calculator
                )
                
                # Update loss breakdown for logging
                for k, v in loss_dict.items():
                    if k != 'total':
                        epoch_loss_breakdown[k] += v
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
        loss_curve.append(avg_loss)
        
        # Print loss breakdown if using combined loss
        if use_combined_loss and num_batches > 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
            print(f"  RF Loss: {epoch_loss_breakdown['rf']/num_batches:.6f}, " +
                  f"L1 Loss: {epoch_loss_breakdown['l1']/num_batches:.6f}, " +
                  f"SSIM Loss: {epoch_loss_breakdown['ssim']/num_batches:.6f}, " +
                  f"SSIM Value: {1-epoch_loss_breakdown['ssim_value']/num_batches:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        # Save checkpoint if output directory is provided
        if output_dir is not None:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pt')
            try:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': rf.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
            except Exception as e:
                print(f"Error saving checkpoint: {str(e)}")
    
    return rf, loss_curve

if __name__ == "__main__":
    # Simple test to ensure the model works
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MonaiRectifiedFlow(img_size=256, in_channels=1, out_channels=1).to(device)
    
    # Test input
    x = torch.randn(2, 1, 256, 256).to(device)
    t = torch.rand(2).to(device)
    
    # Test forward pass
    output = model(x, t)
    print(f"Input shape: {x.shape}, Time shape: {t.shape}, Output shape: {output.shape}")
    
    # Test ODE solver
    rf = RectifiedFlowODE(model, num_steps=10)
    trajectories = rf.sample_ode(x)
    print(f"Number of trajectory steps: {len(trajectories)}")
    print(f"First trajectory shape: {trajectories[0].shape}, Last trajectory shape: {trajectories[-1].shape}")
    
    # Test loss function
    target = torch.randn(2, 1, 256, 256).to(device)
    combined_loss, loss_dict = calculate_combined_loss(output, x, target, t)
    print(f"Combined loss: {combined_loss.item()}, RF: {loss_dict['rf']}, L1: {loss_dict['l1']}, SSIM: {loss_dict['ssim']}") 