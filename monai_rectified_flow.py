import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from tqdm import tqdm

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
            dimensions=2,  # 2D UNet
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

def train_monai_rectified_flow(model, optimizer, source_loader, target_loader, device, epochs):
    """
    Train the Rectified Flow model.
    
    Args:
        model: MonaiRectifiedFlow model
        optimizer: Optimizer
        source_loader: DataLoader for source domain (T1)
        target_loader: DataLoader for target domain (T2)
        device: Device for training
        epochs: Number of epochs
        
    Returns:
        model: Trained model
        loss_curve: Training loss history
    """
    model.train()
    loss_curve = []
    
    for epoch in range(epochs):
        epoch_losses = []
        
        for (source_batch, target_batch) in tqdm(zip(source_loader, target_loader), 
                                                desc=f"Epoch {epoch+1}/{epochs}",
                                                total=min(len(source_loader), len(target_loader))):
            
            source_batch, target_batch = source_batch.to(device), target_batch.to(device)
            
            # Ensure we have matching batch sizes
            current_batch_size = min(source_batch.size(0), target_batch.size(0))
            source_batch = source_batch[:current_batch_size]
            target_batch = target_batch[:current_batch_size]
            
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
    
    return model, loss_curve

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