import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from dataset import create_data_loaders
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class SimplePretrainedModel(nn.Module):
    """
    Simplified pre-trained model for T1-to-T2 MRI translation.
    Uses ResNet18 backbone with 95% frozen parameters.
    """
    def __init__(self, freeze_percent=0.95):
        super().__init__()
        
        # Load pre-trained ResNet18
        self.backbone = resnet18(pretrained=True)
        
        # Modify first layer to accept grayscale + time input (2 channels)
        original_layer = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            2,                              # Grayscale image + time channel
            64,                             # Same as original output channels
            kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Initialize new layer with pretrained weights for first channel
        with torch.no_grad():
            # Average RGB channels for grayscale
            self.backbone.conv1.weight[:, :1] = original_layer.weight.mean(dim=1, keepdim=True)
            # Initialize time channel with small random values
            self.backbone.conv1.weight[:, 1:] = torch.randn_like(self.backbone.conv1.weight[:, 1:]) * 0.01
            
        # Remove final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Create a simple decoder (upsampling)
        self.decoder = nn.Sequential(
            # Upsample to 14x14
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Upsample to 28x28
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Upsample to 56x56
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Upsample to 112x112
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Upsample to 224x224
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # Final layer
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )
        
        # Freeze most of the backbone
        self._freeze_layers(freeze_percent)
    
    def _freeze_layers(self, freeze_percent):
        """Freeze a percentage of the backbone layers"""
        # Get all parameters
        all_params = list(self.backbone.parameters())
        num_params = len(all_params)
        
        # Calculate how many to freeze
        num_to_freeze = int(num_params * freeze_percent)
        
        # Freeze earlier layers first
        for i, param in enumerate(all_params):
            if i < num_to_freeze:
                param.requires_grad = False
        
        # Count frozen vs trainable parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"Frozen {frozen_params:,}/{total_params:,} parameters ({frozen_params/total_params*100:.1f}%)")
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    def forward(self, x, t):
        """Forward pass with image x and time t"""
        # Create time channel
        time_channel = t.view(-1, 1, 1, 1).expand(-1, 1, x.shape[2], x.shape[3])
        
        # Concatenate input and time channel
        x_t = torch.cat([x, time_channel], dim=1)
        
        # Resize if needed (ResNet expects 224x224)
        if x_t.shape[-2:] != (224, 224):
            x_t = F.interpolate(x_t, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Forward pass through backbone
        features = self.backbone(x_t)
        
        # Decode
        output = self.decoder(features)
        
        # Resize back to input size if needed
        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode='bilinear', align_corners=False)
            
        return output

def quick_train_demo(model, train_loader, device, output_dir, num_steps=50):
    """Very quick training demo for the pre-trained model"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    
    # Create optimizer - smaller learning rate for pre-trained layers
    optimizer = torch.optim.Adam([
        {'params': model.backbone.parameters(), 'lr': 1e-5},
        {'params': model.decoder.parameters(), 'lr': 1e-4}
    ])
    
    # Just do a few steps of training
    model.train()
    losses = []
    
    print(f"Starting quick training for {num_steps} steps...")
    progress_bar = tqdm(range(num_steps))
    
    for step in progress_bar:
        try:
            # Get a batch of data
            source_batch, target_batch = next(iter(train_loader))
            source_batch, target_batch = source_batch.to(device), target_batch.to(device)
            
            # Sample random timesteps
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
            losses.append(loss.item())
            progress_bar.set_description(f"Loss: {loss.item():.6f}")
        except Exception as e:
            print(f"Error in training step: {e}")
            continue
    
    # Generate visualization after training
    model.eval()
    with torch.no_grad():
        # Get a batch for visualization
        source_batch, target_batch = next(iter(train_loader))
        source_batch, target_batch = source_batch.to(device), target_batch.to(device)
        
        # Sample at t=1 to get final result
        t = torch.ones(source_batch.size(0), device=device)
        
        # Forward pass
        velocity = model(source_batch, t)
        generated = source_batch + velocity
        
        # Determine how many samples to show (up to 4, but limited by batch size)
        n_samples = min(4, source_batch.size(0))
        
        # Create visualization
        fig, axes = plt.subplots(3, n_samples, figsize=(4*n_samples, 12))
        
        # Handle the case where n_samples=1
        if n_samples == 1:
            # Single sample case
            # T1 source
            axes[0].imshow(source_batch[0, 0].cpu().numpy(), cmap='gray')
            axes[0].set_title('T1 Source')
            axes[0].axis('off')
            
            # Generated T2
            axes[1].imshow(generated[0, 0].cpu().numpy(), cmap='gray')
            axes[1].set_title('Generated T2')
            axes[1].axis('off')
            
            # True T2
            axes[2].imshow(target_batch[0, 0].cpu().numpy(), cmap='gray')
            axes[2].set_title('True T2')
            axes[2].axis('off')
        else:
            # Multiple samples case
            for i in range(n_samples):
                # T1 source
                axes[0, i].imshow(source_batch[i, 0].cpu().numpy(), cmap='gray')
                axes[0, i].set_title('T1 Source')
                axes[0, i].axis('off')
                
                # Generated T2
                axes[1, i].imshow(generated[i, 0].cpu().numpy(), cmap='gray')
                axes[1, i].set_title('Generated T2')
                axes[1, i].axis('off')
                
                # True T2
                axes[2, i].imshow(target_batch[i, 0].cpu().numpy(), cmap='gray')
                axes[2, i].set_title('True T2')
                axes[2, i].axis('off')
        
        # Save the plot
        vis_path = os.path.join(output_dir, 'visualizations', 'results.png')
        plt.tight_layout()
        plt.savefig(vis_path)
        print(f"Saved visualization to {vis_path}")
        
        # Plot loss curve
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
        print(f"Saved loss curve to {os.path.join(output_dir, 'loss_curve.png')}")
    
    print("Quick training demo completed!")
    return model

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Quick Demo of Pretrained MRI Translation')
    parser.add_argument('--t1_dir', type=str, required=True, help='Directory containing T1 slices')
    parser.add_argument('--t2_dir', type=str, required=True, help='Directory containing T2 slices')
    parser.add_argument('--freeze_percent', type=float, default=0.95, help='Percentage of backbone to freeze')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--output_dir', type=str, default='./demo_results', help='Output directory')
    parser.add_argument('--num_steps', type=int, default=50, help='Number of training steps')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, _ = create_data_loaders(
        args.t1_dir,
        args.t2_dir,
        batch_size=args.batch_size,
        train_ratio=0.8,
        num_workers=0  # No parallelism for simple demo
    )
    
    print(f"Loaded {len(train_loader.dataset)} training samples")
    
    # Create model
    model = SimplePretrainedModel(freeze_percent=args.freeze_percent).to(device)
    
    # Train model
    quick_train_demo(
        model,
        train_loader,
        device,
        args.output_dir,
        num_steps=args.num_steps
    ) 