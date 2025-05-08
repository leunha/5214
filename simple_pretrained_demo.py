import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from dataset import create_data_loaders # Assuming this is your refined dataset loader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse # Added for __main__
from datetime import datetime # Added for output_dir timestamp

# Assuming evaluate_model from evaluate_monai.py is adapted or we'll simplify eval here
# from evaluate_monai import evaluate_model
# For this self-contained debug, let's define a simple eval if evaluate_monai.py is complex to adapt now
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

class SimplePretrainedModel(nn.Module):
    """
    Simplified pre-trained model for T1-to-T2 MRI translation.
    Uses ResNet18 backbone. Learns the velocity v = x1 - x0.
    Inference is a single step: x1_pred = x0 + model(x0, t=0).
    """
    def __init__(self, freeze_percent=0.95, img_size=256): # Added img_size
        super().__init__()
        self.img_size = img_size # Store img_size

        # Load pre-trained ResNet18
        self.backbone = resnet18(weights='IMAGENET1K_V1' if True else None) # Explicitly use weights argument

        # Modify first layer to accept grayscale + time input (2 channels)
        original_layer = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            2,  # Grayscale image + time channel
            64,
            kernel_size=7, stride=2, padding=3, bias=False
        )

        # Initialize new layer with pretrained weights for first channel
        with torch.no_grad():
            # Average RGB channels for grayscale for the image channel
            self.backbone.conv1.weight[:, 0:1, :, :] = original_layer.weight.mean(dim=1, keepdim=True)
            # Initialize time channel with small random values
            self.backbone.conv1.weight[:, 1:2, :, :] = torch.randn_like(self.backbone.conv1.weight[:, 1:2, :, :]) * 0.01

        # Remove final classification layer and avgpool
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        # Create a simple decoder (upsampling to match ResNet18 output to img_size)
        # ResNet18 layer4 output is 512 channels, H/32, W/32
        # If img_size is 256, H/32 = 8. So features are [B, 512, 8, 8]
        # We need to upsample 8x8 to 256x256 (5 steps of x2 upsampling)
        self.decoder = nn.Sequential(
            # From 8x8 to 16x16
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # H/16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # From 16x16 to 32x32
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # H/8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # From 32x32 to 64x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # H/4
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # From 64x64 to 128x128
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # H/2
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # From 128x128 to 256x256
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),    # H
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # Final layer to output 1 channel
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            # nn.Tanh() # Output velocity, so Tanh might restrict range too much.
                      # If images are [0,1], velocity is [-1,1].
                      # Let's remove Tanh for now, can be added if outputs are unstable.
        )

        self._freeze_layers(freeze_percent)

    def _freeze_layers(self, freeze_percent):
        all_params = list(self.backbone.parameters())
        num_params_backbone = sum(p.numel() for p in all_params) # p is correct here in sum()
        num_to_freeze_backbone = int(num_params_backbone * freeze_percent)

        frozen_count = 0
        for param in all_params: # The loop variable is 'param'
            if frozen_count < num_to_freeze_backbone:
                param.requires_grad = False
                frozen_count += param.numel() # CORRECTED: was p.numel()
            else:
                # Optionally ensure rest are trainable if unfreezing dynamically
                # Though for a fixed freeze_percent at init, this else might not be strictly needed
                # if all params start as requires_grad=True by default.
                # However, explicitly setting it is safer if the state could be mixed.
                param.requires_grad = True 

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"SimplePretrainedModel: Total params: {total_params:,}")
        print(f"Backbone params: {num_params_backbone:,}. Aimed to freeze: {num_to_freeze_backbone:,} ({freeze_percent*100:.1f}%)")
        # To report actual frozen parameters from the backbone:
        actual_frozen_backbone_params = sum(p.numel() for p in self.backbone.parameters() if not p.requires_grad)
        print(f"Actual frozen backbone params: {actual_frozen_backbone_params:,}")
        print(f"Total Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")

    def forward(self, x, t):
        # Create time channel, x is expected to be [B, 1, H, W]
        time_channel = t.view(-1, 1, 1, 1).expand(-1, 1, x.shape[2], x.shape[3])
        x_t = torch.cat([x, time_channel], dim=1) # Shape [B, 2, H, W]

        # ResNet expects larger inputs if using original strides, but our modified conv1 handles it.
        # No explicit F.interpolate needed if self.img_size matches what the decoder produces
        # after the backbone's downsampling.

        features = self.backbone(x_t)
        output_velocity = self.decoder(features)

        # Ensure output size matches input spatial dimensions, just in case
        if output_velocity.shape[-2:] != x.shape[-2:]:
            output_velocity = F.interpolate(output_velocity, size=x.shape[-2:], mode='bilinear', align_corners=False)

        return output_velocity

def quick_train_demo(model, train_loader, val_loader, device, output_dir, num_epochs=5, img_size=256):
    os.makedirs(output_dir, exist_ok=True)
    vis_output_dir = os.path.join(output_dir, 'visualizations_simple_demo')
    os.makedirs(vis_output_dir, exist_ok=True)

    optimizer = torch.optim.Adam([
        {'params': model.backbone.parameters(), 'lr': 1e-5}, # Slower LR for backbone
        {'params': model.decoder.parameters(), 'lr': 1e-4}
    ])
    
    train_losses = []
    val_losses = []

    print(f"Starting quick training for SimplePretrainedModel for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [T]")
        for source_batch, target_batch in progress_bar:
            source_batch, target_batch = source_batch.to(device), target_batch.to(device)
            
            t = torch.rand(source_batch.size(0), device=device) # Sample t from U(0,1)
            x_t = source_batch * (1 - t.view(-1, 1, 1, 1)) + target_batch * t.view(-1, 1, 1, 1)
            
            optimizer.zero_grad()
            pred_velocity = model(x_t, t) # Model predicts velocity
            
            true_velocity = target_batch - source_batch
            loss = F.mse_loss(pred_velocity, true_velocity)
            
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            progress_bar.set_postfix({"train_loss": loss.item()})
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [V]")
            for source_batch_val, target_batch_val in progress_bar_val:
                source_batch_val, target_batch_val = source_batch_val.to(device), target_batch_val.to(device)
                t_val = torch.rand(source_batch_val.size(0), device=device)
                x_t_val = source_batch_val * (1-t_val.view(-1,1,1,1)) + target_batch_val * t_val.view(-1,1,1,1)
                
                pred_velocity_val = model(x_t_val, t_val)
                true_velocity_val = target_batch_val - source_batch_val
                val_loss_item = F.mse_loss(pred_velocity_val, true_velocity_val)
                epoch_val_loss += val_loss_item.item()
                progress_bar_val.set_postfix({"val_loss": val_loss_item.item()})

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        # Generate visualization after each epoch
        model.eval()
        with torch.no_grad():
            vis_source, vis_target = next(iter(val_loader)) # Use a batch from val_loader for consistency
            vis_source, vis_target = vis_source[:4].to(device), vis_target[:4].to(device) # Max 4 samples
            
            # For inference with this model: input x0 (source_batch) and t=0
            # The model should have learned v = x1 - x0
            # So, predicted x1 = x0 + model(x0, t=0)
            # Or, more aligned with RF training, model(xt,t) predicts v for that xt.
            # For a 1-step generation from x0: x1_pred = x0 + model(x0, t=0) if t=0 is meaningful,
            # or integrate from t=0 to t=1: x1_pred = x0 + model( (x0*(1-t_inf) + x1_guess*t_inf), t_inf )*dt_inf
            # Let's use the simplest: assume model(x0, t=0) gives the full velocity to x1
            
            # For visualization, let's generate T2 from T1 using the model
            # The model is trained to predict (target - source) given (interpolated_xt, t)
            # To generate T2 from T1 (source): predicted_velocity = model(source, t=0) or t=anything if not t-dependent
            # For this simplified model, t=0 is a good choice for inference from source.
            t_inference = torch.zeros(vis_source.size(0), device=device) # Use t=0 for inference
            predicted_velocity_vis = model(vis_source, t_inference)
            generated_t2 = vis_source + predicted_velocity_vis
            generated_t2 = torch.clamp(generated_t2, 0, 1) # Clamp output to image range

            n_samples = vis_source.size(0)
            fig, axes = plt.subplots(3, n_samples, figsize=(4 * n_samples, 9)) # Adjusted figsize
            if n_samples == 1: axes = axes.reshape(3,1) # Ensure axes is 2D

            for i in range(n_samples):
                axes[0, i].imshow(vis_source[i, 0].cpu().numpy(), cmap='gray')
                axes[0, i].set_title('T1 Source')
                axes[0, i].axis('off')
                
                axes[1, i].imshow(generated_t2[i, 0].cpu().numpy(), cmap='gray')
                axes[1, i].set_title('Generated T2')
                axes[1, i].axis('off')
                
                axes[2, i].imshow(vis_target[i, 0].cpu().numpy(), cmap='gray')
                axes[2, i].set_title('True T2')
                axes[2, i].axis('off')
            
            vis_path = os.path.join(vis_output_dir, f'results_epoch_{epoch+1}.png')
            plt.tight_layout()
            plt.savefig(vis_path)
            plt.close(fig)
            print(f"Saved visualization to {vis_path}")
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title('SimplePretrainedModel Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_curve_simple_demo.png'))
    plt.close()
    print(f"Saved loss curve to {os.path.join(output_dir, 'loss_curve_simple_demo.png')}")
    
    print("Quick training demo for SimplePretrainedModel completed!")
    return model

def simple_evaluate_model(model, test_loader, device, output_dir):
    """Simplified evaluation for SimplePretrainedModel."""
    model.eval()
    all_psnr = []
    all_ssim = []
    os.makedirs(output_dir, exist_ok=True)
    print(f"Evaluating SimplePretrainedModel, results in {output_dir}")

    with torch.no_grad():
        for i, (source_batch, target_batch) in enumerate(tqdm(test_loader, desc="Simple Evaluating")):
            source_batch, target_batch = source_batch.to(device), target_batch.to(device)
            
            t_inference = torch.zeros(source_batch.size(0), device=device)
            predicted_velocity = model(source_batch, t_inference)
            generated_batch = source_batch + predicted_velocity
            generated_batch = torch.clamp(generated_batch, 0, 1)

            for j in range(source_batch.size(0)):
                real_img = target_batch[j, 0].cpu().numpy()
                fake_img = generated_batch[j, 0].cpu().numpy()
                
                all_psnr.append(psnr(real_img, fake_img, data_range=1.0))
                all_ssim.append(ssim(real_img, fake_img, data_range=1.0, win_size=7)) # win_size for ssim

            if i == 0: # Visualize first batch
                n_samples = min(4, source_batch.size(0))
                fig, axes = plt.subplots(3, n_samples, figsize=(4 * n_samples, 9))
                if n_samples == 1: axes = axes.reshape(3,1)

                for k in range(n_samples):
                    axes[0, k].imshow(source_batch[k, 0].cpu().numpy(), cmap='gray')
                    axes[0, k].set_title('T1 Source')
                    axes[0, k].axis('off')
                    axes[1, k].imshow(generated_batch[k, 0].cpu().numpy(), cmap='gray')
                    axes[1, k].set_title('Generated T2')
                    axes[1, k].axis('off')
                    axes[2, k].imshow(target_batch[k, 0].cpu().numpy(), cmap='gray')
                    axes[2, k].set_title('True T2')
                    axes[2, k].axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "simple_eval_samples.png"))
                plt.close(fig)

    avg_psnr = np.mean(all_psnr)
    avg_ssim = np.mean(all_ssim)
    print(f"Simple Evaluation Results: Avg PSNR: {avg_psnr:.4f}, Avg SSIM: {avg_ssim:.4f}")
    with open(os.path.join(output_dir, "simple_eval_summary.txt"), "w") as f:
        f.write(f"Avg PSNR: {avg_psnr:.4f}\n")
        f.write(f"Avg SSIM: {avg_ssim:.4f}\n")
    return {"psnr": avg_psnr, "ssim": avg_ssim}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quick Demo of Pretrained MRI Translation (Simple)')
    parser.add_argument('--t1_dir', type=str, required=True, help='Directory containing T1 slices')
    parser.add_argument('--t2_dir', type=str, required=True, help='Directory containing T2 slices')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Training ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Test ratio')
    parser.add_argument('--freeze_percent', type=float, default=0.95, help='Percentage of backbone to freeze')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--img_size', type=int, default=256, help='Image size (must be power of 2 for ResNet downsampling)')
    parser.add_argument('--output_dir', type=str, default='./demo_results_simple', help='Output directory')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of training epochs') # Changed from num_steps
    parser.add_argument('--device', type=str, default='cpu', help='Device: cuda or cpu')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")

    # Create a timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.output_dir, f"run_simple_{timestamp}")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Outputting to: {args.output_dir}")
    
    train_loader, val_loader, test_loader = create_data_loaders(
        args.t1_dir, args.t2_dir,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio,
        num_workers=0 if device.type == 'cpu' else 2 # Adjust num_workers
    )
    
    print(f"Loaded {len(train_loader.dataset)} training, {len(val_loader.dataset)} val, {len(test_loader.dataset)} test samples")
    
    model = SimplePretrainedModel(freeze_percent=args.freeze_percent, img_size=args.img_size).to(device)
    
    model = quick_train_demo(
        model, train_loader, val_loader, device, args.output_dir, num_epochs=args.num_epochs, img_size=args.img_size
    )

    # Simplified evaluation
    eval_output_dir = os.path.join(args.output_dir, "evaluation_simple")
    simple_evaluate_model(model, test_loader, device, eval_output_dir)
    
    # If you want to use the more complex evaluate_model from evaluate_monai.py,
    # you'd need to ensure it can handle this SimplePretrainedModel type.
    # For now, the simple_evaluate_model provides basic metrics.
    # from evaluate_monai import evaluate_model # Make sure this is importable
    # evaluate_model(
    #     model, # This is the tricky part - evaluate_model expects a certain interface
    #     test_loader,
    #     device,
    #     num_steps=1, # As it's a 1-step model
    #     output_dir=os.path.join(args.output_dir, "evaluation_monai_simple_1step")
    # )