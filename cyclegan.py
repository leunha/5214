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
from monai.networks.nets import UNet
from monai.networks.layers import Norm

from dataset import MRISliceDataset, create_data_loaders

class Generator(nn.Module):
    """Residual block for the generator"""
    def __init__(self, in_channels=1,out_channels=1,features=(32,64, 128, 256)):
        super(Generator, self).__init__()
        
        self.unet = UNet(
            spatial_dims=2,  # 2D UNet
            in_channels=in_channels,  # +1 for time channel
            out_channels=out_channels,
            channels=features,
            strides=(2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH
        )
    
    def forward(self, x):
        # Forward pass through UNet
        return self.unet(x)

class Discriminator(nn.Module):
    """Discriminator network for CycleGAN"""
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, normalize=True, activation=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            if activation:
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(in_channels, 32, normalize=False),
            *discriminator_block(32, 64),
            *discriminator_block(64, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 256),
            *discriminator_block(256, 1,normalize=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

class CycleGAN:
    """CycleGAN implementation for T1-T2 MRI translation"""
    def __init__(self, device, lambda_cycle=10.0, lambda_identity=5.0,in_channels=1,out_channels=1,features=(64, 128, 256)):
        self.device = device
        
        # Use MONAI's UNet as generator backbone
        self.G_T1_to_T2 = Generator().to(device)
        self.G_T2_to_T1 = Generator().to(device)
        self.D_T1 = Discriminator().to(device)
        self.D_T2 = Discriminator().to(device)
        
        # Initialize optimizers
        self.optimizer_G = optim.Adam(
            list(self.G_T1_to_T2.parameters()) + list(self.G_T2_to_T1.parameters()),
            lr=0.0001, betas=(0.5, 0.999)
        )
        self.optimizer_D = optim.Adam(
            list(self.D_T1.parameters()) + list(self.D_T2.parameters()),
            lr=0.0001, betas=(0.5, 0.999)
        )
        
        # Loss functions
        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        
        # Loss weights
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.G_T1_to_T2.train()
        self.G_T2_to_T1.train()
        self.D_T1.train()
        self.D_T2.train()
        
        epoch_losses = {
            'G': [],
            'D': [],
            'cycle': [],
            'identity': []
        }
        
        for t1, t2 in tqdm(train_loader, desc="Training"):
            # Move data to device
            t1 = t1.to(self.device)
            t2 = t2.to(self.device)
            
            # Generate fake images
            fake_t2 = self.G_T1_to_T2(t1)
            fake_t1 = self.G_T2_to_T1(t2)
            
            # Reconstruct original images
            reconstructed_t1 = self.G_T2_to_T1(fake_t2)
            reconstructed_t2 = self.G_T1_to_T2(fake_t1)
            
            # Identity mapping
            identity_t1 = self.G_T2_to_T1(t1)
            identity_t2 = self.G_T1_to_T2(t2)
            
            # Adversarial loss
            valid = torch.ones(t1.size(0), 1, 1, 1, device=self.device)
            fake = torch.zeros(t1.size(0), 1, 1, 1, device=self.device)
            
            # Generator loss
            loss_GAN_T1_to_T2 = self.criterion_GAN(self.D_T2(fake_t2), valid)
            loss_GAN_T2_to_T1 = self.criterion_GAN(self.D_T1(fake_t1), valid)
            loss_GAN = (loss_GAN_T1_to_T2 + loss_GAN_T2_to_T1) / 2
            
            # Cycle loss
            loss_cycle_T1 = self.criterion_cycle(reconstructed_t1, t1)
            loss_cycle_T2 = self.criterion_cycle(reconstructed_t2, t2)
            loss_cycle = (loss_cycle_T1 + loss_cycle_T2) / 2
            
            # Identity loss
            loss_identity_T1 = self.criterion_identity(identity_t1, t1)
            loss_identity_T2 = self.criterion_identity(identity_t2, t2)
            loss_identity = (loss_identity_T1 + loss_identity_T2) / 2
            
            # Total generator loss
            loss_G = loss_GAN + self.lambda_cycle * loss_cycle + self.lambda_identity * loss_identity
            
            # Discriminator loss
            loss_D_T1 = (self.criterion_GAN(self.D_T1(t1), valid) + 
                        self.criterion_GAN(self.D_T1(fake_t1.detach()), fake)) / 2
            loss_D_T2 = (self.criterion_GAN(self.D_T2(t2), valid) + 
                        self.criterion_GAN(self.D_T2(fake_t2.detach()), fake)) / 2
            loss_D = (loss_D_T1 + loss_D_T2) / 2
            
            # Update generators
            self.optimizer_G.zero_grad()
            loss_G.backward()
            self.optimizer_G.step()
            
            # Update discriminators
            self.optimizer_D.zero_grad()
            loss_D.backward()
            self.optimizer_D.step()
            
            # Record losses
            epoch_losses['G'].append(loss_G.item())
            epoch_losses['D'].append(loss_D.item())
            epoch_losses['cycle'].append(loss_cycle.item())
            epoch_losses['identity'].append(loss_identity.item())
        
        # Calculate average losses
        avg_losses = {k: sum(v) / len(v) for k, v in epoch_losses.items()}
        return avg_losses
    
    def save_checkpoint(self, epoch, output_dir):
        """Save model checkpoint"""
        os.makedirs(output_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'G_T1_to_T2_state_dict': self.G_T1_to_T2.state_dict(),
            'G_T2_to_T1_state_dict': self.G_T2_to_T1.state_dict(),
            'D_T1_state_dict': self.D_T1.state_dict(),
            'D_T2_state_dict': self.D_T2.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict()
        }
        
        torch.save(checkpoint, os.path.join(output_dir, f'cyclegan_epoch_{epoch}.pt'))
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        
        self.G_T1_to_T2.load_state_dict(checkpoint['G_T1_to_T2_state_dict'])
        self.G_T2_to_T1.load_state_dict(checkpoint['G_T2_to_T1_state_dict'])
        self.D_T1.load_state_dict(checkpoint['D_T1_state_dict'])
        self.D_T2.load_state_dict(checkpoint['D_T2_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        
        return checkpoint['epoch']
    
    def evaluate(self, data_loader):
        """Evaluate model performance"""
        self.G_T1_to_T2.eval()
        self.G_T2_to_T1.eval()
        
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
                
                # Generate T2 from T1
                fake_t2 = self.G_T1_to_T2(t1)
                
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
        self.G_T1_to_T2.eval()
        self.G_T2_to_T1.eval()
        
        os.makedirs(os.path.join(output_dir, 'evaluation'), exist_ok=True)
        
        # Get one batch
        t1, t2 = next(iter(data_loader))
        t1, t2 = t1.to(self.device), t2.to(self.device)
        
        with torch.no_grad():
            # Generate translations
            fake_t2 = self.G_T1_to_T2(t1)
            fake_t1 = self.G_T2_to_T1(t2)
            
            # Create figure
            n_samples = min(4, t1.size(0))
            fig, axes = plt.subplots(4, n_samples, figsize=(4*n_samples, 12))
            
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
                
                # Show T2
                ax_row[2].imshow(t2[i, 0].cpu().numpy(), cmap='gray')
                ax_row[2].set_title('T2')
                ax_row[2].axis('off')
                
                # Show generated T1
                ax_row[3].imshow(fake_t1[i, 0].cpu().numpy(), cmap='gray')
                ax_row[3].set_title('Generated T1')
                ax_row[3].axis('off')
            
            # Add row labels
            if n_samples > 0:
                if n_samples == 1:
                    axes[0].set_ylabel("T1", fontsize=12)
                    axes[1].set_ylabel("Generated T2", fontsize=12)
                    axes[2].set_ylabel("T2", fontsize=12)
                    axes[3].set_ylabel("Generated T1", fontsize=12)
                else:
                    axes[0, 0].set_ylabel("T1", fontsize=12)
                    axes[1, 0].set_ylabel("Generated T2", fontsize=12)
                    axes[2, 0].set_ylabel("T2", fontsize=12)
                    axes[3, 0].set_ylabel("Generated T1", fontsize=12)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'evaluation', 'cyclegan_results.png'))
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
    model.G_T1_to_T2.eval()
    model.G_T2_to_T1.eval()
    
    all_metrics = []
    visualization_done = False
    
    with torch.no_grad():
        for i, (source_batch, target_batch) in enumerate(tqdm(test_loader, desc="Evaluating")):
            if test_samples is not None and i >= test_samples:
                break
                
            source_batch, target_batch = source_batch.to(device), target_batch.to(device)
            
            # Time the generation process
            start_time = time.time()
            generated_batch = model.G_T1_to_T2(source_batch)
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
    
    # Create CycleGAN model
    model = CycleGAN(
        device=args.device,
        lambda_cycle=args.lambda_cycle,
        lambda_identity=args.lambda_identity
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
            'D': [],
            'cycle': [],
            'identity': []
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
            logging.info(f"  Cycle Loss: {train_losses['cycle']:.6f}")
            logging.info(f"  Identity Loss: {train_losses['identity']:.6f}")
            
            # Record losses
            loss_history['G'].append(train_losses['G'])
            loss_history['D'].append(train_losses['D'])
            loss_history['cycle'].append(train_losses['cycle'])
            loss_history['identity'].append(train_losses['identity'])
            
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
        plt.figure(figsize=(12, 8))
        
        # Plot generator and discriminator losses
        plt.subplot(2, 1, 1)
        plt.plot(loss_history['G'], label='Generator Loss')
        plt.plot(loss_history['D'], label='Discriminator Loss')
        plt.title('Adversarial Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot cycle and identity losses
        plt.subplot(2, 1, 2)
        plt.plot(loss_history['cycle'], label='Cycle Loss')
        plt.plot(loss_history['identity'], label='Identity Loss')
        plt.title('Cycle and Identity Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
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
    parser = argparse.ArgumentParser(description='Train CycleGAN for T1-T2 MRI Translation')
    
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
    
    # CycleGAN arguments
    parser.add_argument('--lambda_cycle', type=float, default=1.0, help='Weight for cycle consistency loss')
    parser.add_argument('--lambda_identity', type=float, default=1.0, help='Weight for identity loss')
    
    # Evaluation arguments
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model instead of training')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint for evaluation')
    parser.add_argument('--max_test_batches', type=int, default=None, help='Maximum number of batches to test')
    
    args = parser.parse_args()
    main(args)
