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
import logging
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
import pandas as pd

from dataset import MRISliceDataset, create_data_loaders

class Generator(nn.Module):
    """Generator network for DCGAN"""
    def __init__(self, latent_dim=100, img_channels=1, features_g=64):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Initial size: 4x4
        self.initial_size = 4
        
        # Calculate the initial number of channels
        initial_channels = features_g * 8
        
        # Project and reshape
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, initial_channels * self.initial_size * self.initial_size),
            nn.BatchNorm1d(initial_channels * self.initial_size * self.initial_size),
            nn.ReLU(True)
        )
        
        # Transposed convolution layers
        self.main = nn.Sequential(
            # Input: 4x4
            nn.ConvTranspose2d(initial_channels, features_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(True),
            
            # 8x8
            nn.ConvTranspose2d(features_g * 4, features_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 2),
            nn.ReLU(True),
            
            # 16x16
            nn.ConvTranspose2d(features_g * 2, features_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g),
            nn.ReLU(True),
            
            # 32x32
            nn.ConvTranspose2d(features_g, features_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g),
            nn.ReLU(True),
            
            # 64x64
            nn.ConvTranspose2d(features_g, features_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g),
            nn.ReLU(True),
            
            # 128x128
            nn.ConvTranspose2d(features_g, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: 256x256
        )
    
    def forward(self, z):
        # Project and reshape
        x = self.projection(z)
        x = x.view(-1, self.main[0].in_channels, self.initial_size, self.initial_size)
        
        # Generate image
        return self.main(x)

class Discriminator(nn.Module):
    """Discriminator network for DCGAN"""
    def __init__(self, img_channels=1, features_d=64):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            # Input: 256x256
            nn.Conv2d(img_channels, features_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128x128
            nn.Conv2d(features_d, features_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64x64
            nn.Conv2d(features_d * 2, features_d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32
            nn.Conv2d(features_d * 4, features_d * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16
            nn.Conv2d(features_d * 8, features_d * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 16),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8
            nn.Conv2d(features_d * 16, features_d * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 32),
            nn.LeakyReLU(0.2, inplace=True),

            # 4x4
            nn.Conv2d(features_d * 32, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # Output: 1x1
        )
    
    def forward(self, x):
        return self.main(x)

class DCGAN:
    """DCGAN implementation for T1-T2 MRI translation"""
    def __init__(self, device, latent_dim=100, lr=0.0002, beta1=0.5):
        self.device = device
        self.latent_dim = latent_dim
        
        # Initialize generator and discriminator
        self.G = Generator(latent_dim=latent_dim).to(device)
        self.D = Discriminator().to(device)
        
        # Initialize optimizers
        self.optimizer_G = optim.Adam(self.G.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizer_D = optim.Adam(self.D.parameters(), lr=lr, betas=(beta1, 0.999))
        
        # Loss function
        self.criterion = nn.BCELoss()
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.G.train()
        self.D.train()
        
        epoch_losses = {
            'G': [],
            'D': []
        }
        
        for t1, t2 in tqdm(train_loader, desc="Training"):
            # Move data to device
            t1 = t1.to(self.device)
            t2 = t2.to(self.device)
            
            batch_size = t1.size(0)
            
            # Create labels
            real_label = torch.ones(batch_size, 1, 1, 1, device=self.device)
            fake_label = torch.zeros(batch_size, 1, 1, 1, device=self.device)
            
            # Train Discriminator
            self.optimizer_D.zero_grad()
            
            # Real loss
            output_real = self.D(t2)
            loss_D_real = self.criterion(output_real, real_label)
            
            # Fake loss
            noise = torch.randn(batch_size, self.latent_dim, device=self.device)
            fake_t2 = self.G(noise)
            output_fake = self.D(fake_t2.detach())
            loss_D_fake = self.criterion(output_fake, fake_label)
            
            # Total discriminator loss
            loss_D = loss_D_real + loss_D_fake
            loss_D.backward()
            self.optimizer_D.step()
            
            # Train Generator
            self.optimizer_G.zero_grad()
            
            # Generate fake images
            noise = torch.randn(batch_size, self.latent_dim, device=self.device)
            fake_t2 = self.G(noise)
            
            # Generator loss
            output = self.D(fake_t2)
            loss_G = self.criterion(output, real_label)
            loss_G.backward()
            self.optimizer_G.step()
            
            # Record losses
            epoch_losses['G'].append(loss_G.item())
            epoch_losses['D'].append(loss_D.item())
        
        # Calculate average losses
        avg_losses = {k: sum(v) / len(v) for k, v in epoch_losses.items()}
        return avg_losses
    
    def save_checkpoint(self, epoch, output_dir):
        """Save model checkpoint"""
        os.makedirs(output_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'G_state_dict': self.G.state_dict(),
            'D_state_dict': self.D.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict()
        }
        
        torch.save(checkpoint, os.path.join(output_dir, f'dcgan_epoch_{epoch}.pt'))
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        
        self.G.load_state_dict(checkpoint['G_state_dict'])
        self.D.load_state_dict(checkpoint['D_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        
        return checkpoint['epoch']
    
    def evaluate(self, data_loader):
        """Evaluate model performance"""
        self.G.eval()
        
        all_metrics = {
            'ssim': [],
            'psnr': [],
            'mse': []
        }
        
        with torch.no_grad():
            for t1, t2 in tqdm(data_loader, desc="Evaluating"):
                # Move data to device
                t1 = t1.to(self.device)
                t2 = t2.to(self.device)
                
                # Generate T2 from random noise
                noise = torch.randn(t1.size(0), self.latent_dim, device=self.device)
                fake_t2 = self.G(noise)
                
                # Calculate metrics
                metrics = self._calculate_metrics(t2, fake_t2)
                
                # Accumulate metrics
                for k, v in metrics.items():
                    all_metrics[k].append(v)
        
        # Calculate average metrics
        avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
        return avg_metrics
    
    def _calculate_metrics(self, real, fake):
        """Calculate evaluation metrics between real and generated images"""
        metrics = {}
        
        # Convert to numpy arrays
        real = real.cpu().numpy()
        fake = fake.cpu().numpy()
        
        # Calculate metrics for each image in the batch
        metrics['ssim'] = np.mean([ssim(r[0], f[0], data_range=1.0) for r, f in zip(real, fake)])
        metrics['psnr'] = np.mean([psnr(r[0], f[0], data_range=1.0) for r, f in zip(real, fake)])
        metrics['mse'] = np.mean([mse(r[0], f[0]) for r, f in zip(real, fake)])
        
        return metrics
    
    def visualize_results(self, data_loader, output_dir):
        """Visualize evaluation results"""
        self.G.eval()
        
        os.makedirs(os.path.join(output_dir, 'evaluation'), exist_ok=True)
        
        # Get one batch
        t1, t2 = next(iter(data_loader))
        t1, t2 = t1.to(self.device), t2.to(self.device)
        
        with torch.no_grad():
            # Generate random noise
            noise = torch.randn(t1.size(0), self.latent_dim, device=self.device)
            
            # Generate fake T2 images
            fake_t2 = self.G(noise)
            
            # Create figure
            n_samples = min(4, t1.size(0))
            fig, axes = plt.subplots(3, n_samples, figsize=(4*n_samples, 9))
            
            for i in range(n_samples):
                if n_samples == 1:
                    ax_row = axes
                else:
                    ax_row = axes[:, i]
                
                # Show T1
                ax_row[0].imshow(t1[i, 0].cpu().numpy(), cmap='gray')
                ax_row[0].set_title('T1')
                ax_row[0].axis('off')
                
                # Show generated T2
                ax_row[1].imshow(fake_t2[i, 0].cpu().numpy(), cmap='gray')
                ax_row[1].set_title('Generated T2')
                ax_row[1].axis('off')
                
                # Show real T2
                ax_row[2].imshow(t2[i, 0].cpu().numpy(), cmap='gray')
                ax_row[2].set_title('Real T2')
                ax_row[2].axis('off')
            
            # Add row labels
            if n_samples > 0:
                if n_samples == 1:
                    axes[0].set_ylabel("T1", fontsize=12)
                    axes[1].set_ylabel("Generated T2", fontsize=12)
                    axes[2].set_ylabel("Real T2", fontsize=12)
                else:
                    axes[0, 0].set_ylabel("T1", fontsize=12)
                    axes[1, 0].set_ylabel("Generated T2", fontsize=12)
                    axes[2, 0].set_ylabel("Real T2", fontsize=12)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'evaluation', 'dcgan_results.png'))
            plt.close()

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

def visualize_results(source_batch, generated_batch, target_batch, output_dir, prefix=''):
    """
    Visualize evaluation results
    
    Args:
        source_batch: Batch of source images (T1)
        generated_batch: Batch of generated images (T2)
        target_batch: Batch of target images (true T2)
        output_dir: Directory to save visualizations
        prefix: Prefix for output filenames
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
    plt.savefig(os.path.join(output_dir, f'{prefix}samples.png'))
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
    plt.savefig(os.path.join(output_dir, f'{prefix}difference_maps.png'))
    plt.close()

def evaluate_model(model, test_loader, device, output_dir='./evaluation_results', test_samples=None):
    """
    Evaluate the model on test data
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device: Device to use
        output_dir: Directory to save results
        test_samples: Number of samples to test (None = all)

    Returns:
        pd.DataFrame: Metrics for each test sample
    """
    model.G.eval()
    
    all_metrics = []
    visualization_done = False
    
    with torch.no_grad():
        for i, (source_batch, target_batch) in enumerate(tqdm(test_loader, desc="Evaluating")):
            if test_samples is not None and i >= test_samples:
                break
                
            source_batch, target_batch = source_batch.to(device), target_batch.to(device)
            
            # Time the generation process
            start_time = time.time()
            noise = torch.randn(source_batch.size(0), model.latent_dim, device=device)
            generated_batch = model.G(noise)
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
                    output_dir
                )
                visualization_done = True
    
    # Convert metrics to DataFrame
    df_metrics = pd.DataFrame(all_metrics)
    
    # Save metrics CSV
    os.makedirs(output_dir, exist_ok=True)
    df_metrics.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
    
    # Calculate and print summary statistics
    summary = df_metrics.describe()
    summary.to_csv(os.path.join(output_dir, 'metrics_summary.csv'))
    
    print("\nMetrics Summary:")
    print(summary)
    
    # Create metrics visualizations
    plt.figure(figsize=(10, 6))
    plt.boxplot([df_metrics['psnr'], df_metrics['ssim'], df_metrics['mse'] * 100, df_metrics['mae'] * 100])
    plt.xticks([1, 2, 3, 4], ['PSNR', 'SSIM', 'MSE×100', 'MAE×100'])
    plt.title('Metrics Distribution')
    plt.savefig(os.path.join(output_dir, 'metrics_boxplot.png'))
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
    plt.savefig(os.path.join(output_dir, 'metrics_histograms.png'))
    plt.close()
    
    return df_metrics

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
    
    # Create DCGAN model
    model = DCGAN(
        device=args.device,
        latent_dim=args.latent_dim,
        lr=args.learning_rate,
        beta1=args.beta1
    )
    
    if args.evaluate:
        # Load checkpoint if provided
        if args.checkpoint:
            model.load_checkpoint(args.checkpoint)
            logging.info(f"Loaded checkpoint from {args.checkpoint}")
        
        # Evaluate model
        metrics = evaluate_model(
            model,
            test_loader,
            args.device,
            output_dir=args.output_dir,
            test_samples=args.max_test_batches
        )
    else:
        # Initialize loss history
        loss_history = {
            'G': [],
            'D': []
        }
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(args.epochs):
            # Train for one epoch
            train_losses = model.train_epoch(train_loader)
            
            # Log losses
            logging.info(f"Epoch {epoch+1}/{args.epochs}")
            logging.info(f"  Generator Loss: {train_losses['G']:.6f}")
            logging.info(f"  Discriminator Loss: {train_losses['D']:.6f}")
            
            # Record losses
            loss_history['G'].append(train_losses['G'])
            loss_history['D'].append(train_losses['D'])
            
            # Save checkpoint
            if (epoch + 1) % args.save_interval == 0:
                model.save_checkpoint(epoch + 1, args.output_dir)
                logging.info(f"Saved checkpoint for epoch {epoch + 1}")
        
        total_time = time.time() - start_time
        logging.info(f"Training completed in {total_time/60:.2f} minutes")
        
        # Save final model
        model.save_checkpoint(args.epochs, args.output_dir)
        logging.info(f"Saved final model")
        
        # Plot loss curves
        plt.figure(figsize=(10, 5))
        plt.plot(loss_history['G'], label='Generator Loss')
        plt.plot(loss_history['D'], label='Discriminator Loss')
        plt.title('Training Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(args.output_dir, 'loss_curves.png'))
        plt.close()
        
        # Save loss history
        loss_df = pd.DataFrame(loss_history)
        loss_df.to_csv(os.path.join(args.output_dir, 'loss_history.csv'), index=False)
        
        # Final evaluation on test set
        logging.info("Performing final evaluation on test set...")
        test_metrics = evaluate_model(
            model,
            test_loader,
            args.device,
            output_dir=os.path.join(args.output_dir, 'final_evaluation'),
            test_samples=args.max_test_batches
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DCGAN for T1-T2 MRI Translation')
    
    # Data arguments
    parser.add_argument('--t1_dir', type=str, required=True, help='Directory containing T1 slices')
    parser.add_argument('--t2_dir', type=str, required=True, help='Directory containing T2 slices')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Training ratio for train/val split')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation ratio for train/val split')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Test ratio for train/val split')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda or cpu)')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--save_interval', type=int, default=5, help='Save model every N epochs')
    
    # DCGAN arguments
    parser.add_argument('--latent_dim', type=int, default=100, help='Dimension of latent space')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 parameter for Adam optimizer')
    
    # Evaluation arguments
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model instead of training')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint for evaluation')
    parser.add_argument('--max_test_batches', type=int, default=None, help='Maximum number of batches to test')
    
    args = parser.parse_args()
    main(args)
