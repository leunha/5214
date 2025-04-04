import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from datetime import datetime

from _2d_dataset import MRISliceDataset

class LightweightUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        
        # Reduced number of features for faster training
        self.enc1 = self._make_layer(in_channels, 32)
        self.enc2 = self._make_layer(32, 64)
        self.enc3 = self._make_layer(64, 128)
        
        self.dec3 = self._make_layer(128, 64)
        self.dec2 = self._make_layer(64, 32)
        self.dec1 = self._make_layer(32, out_channels)
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Decoder with skip connections
        d3 = self.dec3(self.upsample(e3))
        d2 = self.dec2(self.upsample(d3 + e2))
        d1 = self.dec1(d2 + e1)
        
        return d1

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for t1_images, t2_images in tqdm(train_loader, desc="Training"):
        t1_images = t1_images.to(device)
        t2_images = t2_images.to(device)
        
        optimizer.zero_grad()
        outputs = model(t1_images)
        loss = criterion(outputs, t2_images)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for t1_images, t2_images in tqdm(val_loader, desc="Validation"):
            t1_images = t1_images.to(device)
            t2_images = t2_images.to(device)
            
            outputs = model(t1_images)
            loss = criterion(outputs, t2_images)
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def visualize_results(model, val_loader, device, epoch, save_dir):
    model.eval()
    
    # Get a batch of validation data
    t1_images, t2_images = next(iter(val_loader))
    
    with torch.no_grad():
        t1_images = t1_images.to(device)
        outputs = model(t1_images)
    
    # Move tensors to CPU and convert to numpy
    t1_images = t1_images.cpu()
    t2_images = t2_images.cpu()
    outputs = outputs.cpu()
    
    # Create visualization
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    for i in range(4):
        # Display T1 input
        axes[0, i].imshow(t1_images[i, 0], cmap='gray')
        axes[0, i].set_title('T1 Input')
        axes[0, i].axis('off')
        
        # Display T2 ground truth
        axes[1, i].imshow(t2_images[i, 0], cmap='gray')
        axes[1, i].set_title('T2 Ground Truth')
        axes[1, i].axis('off')
        
        # Display model output
        axes[2, i].imshow(outputs[i, 0], cmap='gray')
        axes[2, i].set_title('Model Output')
        axes[2, i].axis('off')
    
    plt.suptitle(f'Results after epoch {epoch}')
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'results_epoch_{epoch}.png'))
    plt.close()

def main():
    # Training parameters
    batch_size = 8
    num_epochs = 3
    learning_rate = 0.001
    device = torch.device('cpu')  # Use CPU for laptop training
    
    # Create save directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f'training_results_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    
    # Create dataset and data loaders
    train_dataset = MRISliceDataset(
        t1_dir='./processed_dataset/IXI-T1',
        t2_dir='./processed_dataset/IXI-T2',
        normalize=True
    )
    
    # Split dataset
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # Create model, loss function, and optimizer
    model = LightweightUNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    print(f"Starting training with {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
    print(f"Results will be saved in: {save_dir}")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Visualize results
        visualize_results(model, val_loader, device, epoch+1, save_dir)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }
        torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt'))
    
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'))
    plt.close()

if __name__ == "__main__":
    main()