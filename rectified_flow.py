import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class ImageRectifiedFlow(nn.Module):
    """
    Rectified Flow model for image-to-image translation.
    Uses a U-Net architecture to predict the velocity field.
    """
    def __init__(self, img_size=256, in_channels=1, base_channels=64):
        super().__init__()
        self.img_size = img_size
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )
        
        # Encoder
        self.enc1 = UNetBlock(in_channels + 1, base_channels)  # +1 for time channel
        self.enc2 = UNetBlock(base_channels, base_channels*2)
        self.enc3 = UNetBlock(base_channels*2, base_channels*4)
        self.enc4 = UNetBlock(base_channels*4, base_channels*8)
        
        # Decoder
        self.dec4 = UNetBlock(base_channels*8, base_channels*4)
        self.dec3 = UNetBlock(base_channels*4*2, base_channels*2)
        self.dec2 = UNetBlock(base_channels*2*2, base_channels)
        self.dec1 = UNetBlock(base_channels*2, base_channels)
        
        # Final layer
        self.final = nn.Conv2d(base_channels, in_channels, 1)
        
    def forward(self, x, t):
        # Expand time to match spatial dimensions
        t = t.view(-1, 1, 1, 1).expand(-1, 1, self.img_size, self.img_size)
        x = torch.cat([x, t], dim=1)
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        
        # Decoder
        d4 = self.dec4(F.interpolate(e4, scale_factor=2))
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, scale_factor=2), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2), e1], dim=1))
        
        return self.final(d1)

class RectifiedFlow:
    """
    Rectified Flow class implementing the training and sampling procedures.
    """
    def __init__(self, model, num_steps=100):
        """
        Initialize the Rectified Flow.
        
        Args:
            model: Neural network model to predict velocity field
            num_steps: Number of steps for ODE integration
        """
        self.model = model
        self.num_steps = num_steps
        
    def sample_ode(self, z0, N=None):
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
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        for step in tqdm(range(training_steps)):
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
            
        return self

def train_rectified_flow(model, optimizer, source_loader, target_loader, device, epochs, batch_size):
    loss_curve = []
    
    for epoch in range(epochs):
        epoch_losses = []
        
        for (source_batch, _), (target_batch, _) in zip(source_loader, target_loader):
            source_batch = source_batch.to(device)
            target_batch = target_batch.to(device)
            
            # Ensure we have the correct batch size
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
    
    return model, loss_curve 