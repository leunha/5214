import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import argparse # Added for __main__
from datetime import datetime # Added for output_dir timestamp

# Assuming evaluate_model from evaluate_monai.py is correctly imported and usable
from evaluate_monai import evaluate_model
from dataset import create_data_loaders # Assuming this is your refined dataset loader


class EfficientPretrainedFlow(nn.Module):
    """
    Efficient Rectified Flow model for T1-to-T2 MRI translation using a pre-trained ResNet18
    backbone.
    """
    def __init__(self, img_size=256, in_channels=1, out_channels=1, freeze_ratio=0.95, pretrained=True):
        super().__init__()
        self.img_size = img_size

        resnet = resnet18(weights='IMAGENET1K_V1' if pretrained else None)

        original_conv1 = resnet.conv1
        # Input to conv1 will be image + time channel
        self.conv1_modified = nn.Conv2d(
            in_channels + 1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        if pretrained:
            with torch.no_grad():
                new_weight = self.conv1_modified.weight.clone()
                # Image channel weights from averaged RGB
                new_weight[:, 0:in_channels, :, :] = original_conv1.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
                # Time channel weights initialized randomly
                new_weight[:, in_channels:, :, :] = torch.randn_like(new_weight[:, in_channels:, :, :]) * 0.01
                self.conv1_modified.weight.copy_(new_weight)
        
        # Encoder using ResNet parts
        self.encoder_layers = nn.Sequential(
            # self.conv1_modified, # This will be handled in forward
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        # Freeze encoder layers (excluding the new conv1_modified which is part of self)
        self._freeze_encoder_layers(self.encoder_layers, freeze_ratio)


        # Decoder (same as in simple_pretrained_demo for consistency if img_size=256 and ResNet18 backbone)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1), # Changed kernel to 3, padding 1 for same size
        )
        
        # Report parameters after freezing
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"EfficientPretrainedFlow: Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")

    def _freeze_encoder_layers(self, encoder_sequential, freeze_ratio):
        """Freeze a percentage of parameters within the encoder_sequential module."""
        encoder_params_list = list(encoder_sequential.parameters())
        num_encoder_params = sum(p.numel() for p in encoder_params_list)
        params_to_freeze = int(num_encoder_params * freeze_ratio)
        
        frozen_count = 0
        for param in encoder_params_list:
            if frozen_count < params_to_freeze:
                param.requires_grad = False
                frozen_count += param.numel()
            else:
                param.requires_grad = True # Ensure subsequent layers are trainable
        print(f"EfficientPretrainedFlow Encoder: Total params: {num_encoder_params:,}. Aimed to freeze: {params_to_freeze:,} ({freeze_ratio*100:.1f}%)")
        print(f"Actual frozen encoder params: {frozen_count:,}")


    def forward(self, x, t):
        time_channel = t.view(-1, 1, 1, 1).expand(-1, 1, x.shape[2], x.shape[3])
        x_t = torch.cat([x, time_channel], dim=1)
        
        # Pass through modified conv1 first
        inter_features = self.conv1_modified(x_t)
        # Then through the rest of the encoder
        encoded_features = self.encoder_layers(inter_features)
        
        output_velocity = self.decoder(encoded_features)

        if output_velocity.shape[-2:] != x.shape[-2:]:
            output_velocity = F.interpolate(output_velocity, size=x.shape[-2:], mode='bilinear', align_corners=False)
            
        return output_velocity

class EfficientRectifiedFlowODE:
    def __init__(self, model, num_steps=50):
        self.model = model
        self.num_steps = num_steps
    
    def sample_ode(self, z0, N=None):
        if N is None: N = self.num_steps
        device = z0.device
        trajectories = [z0.clone()] # Store initial state
        z = z0.clone()
        
        dt = 1.0 / N
        self.model.eval() # Ensure model is in eval mode for ODE sampling
        with torch.no_grad():
            for i in range(N):
                t_val = torch.ones(z.shape[0], device=device) * (i * dt)
                dz = self.model(z, t_val)
                z = z + dz * dt
                trajectories.append(z.clone())
        return trajectories

def create_optimizer_efficient(model, base_lr=1e-4):
    # Separate params for conv1_modified, encoder_layers, and decoder
    conv1_params = list(model.conv1_modified.parameters())
    encoder_frozen_params = [p for p in model.encoder_layers.parameters() if not p.requires_grad] # Should be empty if all trainable
    encoder_trainable_params = [p for p in model.encoder_layers.parameters() if p.requires_grad]
    decoder_params = list(model.decoder.parameters())

    param_groups = []
    # Trainable conv1_modified parts (should be all of it)
    if any(p.requires_grad for p in conv1_params):
         param_groups.append({'params': filter(lambda p: p.requires_grad, conv1_params), 'lr': base_lr * 0.1}) # Slightly lower for this adapted layer
    
    if encoder_trainable_params:
         param_groups.append({'params': encoder_trainable_params, 'lr': base_lr * 0.05}) # Lowest for deeper backbone
    
    param_groups.append({'params': decoder_params, 'lr': base_lr})

    optimizer = torch.optim.AdamW(param_groups)
    print(f"Optimizer created with {len(param_groups)} parameter groups.")
    return optimizer

def quick_train_efficient(model, train_loader, val_loader, device, output_dir, 
                         num_epochs=1, early_stop_patience=None, img_size=256, ode_steps_vis=20): # Added img_size
    os.makedirs(output_dir, exist_ok=True)
    vis_output_dir = os.path.join(output_dir, 'visualizations_efficient_rf')
    os.makedirs(vis_output_dir, exist_ok=True)
    
    optimizer = create_optimizer_efficient(model, base_lr=1e-4) # Adjusted default LR
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3# Increased patience
    )
    
    ode_solver = EfficientRectifiedFlowODE(model, num_steps=ode_steps_vis)
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Get a fixed batch for visualization from val_loader
    try:
        val_batch_x_vis, val_batch_y_vis = next(iter(val_loader))
        val_batch_x_vis = val_batch_x_vis[:4].to(device) # Max 4 samples
        val_batch_y_vis = val_batch_y_vis[:4].to(device)
    except StopIteration:
        print("Validation loader is empty, cannot get visualization batch.")
        val_batch_x_vis, val_batch_y_vis = None, None


    print(f"Starting quick training for EfficientPretrainedFlow for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [T]")
        for source_batch, target_batch in progress_bar:
            source_batch, target_batch = source_batch.to(device), target_batch.to(device)
            
            t = torch.rand(source_batch.size(0), device=device)
            x_t = source_batch * (1 - t.view(-1, 1, 1, 1)) + target_batch * t.view(-1, 1, 1, 1)
            
            optimizer.zero_grad()
            pred_velocity = model(x_t, t)
            true_velocity = target_batch - source_batch
            loss = F.mse_loss(pred_velocity, true_velocity)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
            optimizer.step()
            
            epoch_train_loss += loss.item()
            progress_bar.set_postfix({"train_loss": loss.item()})
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [V]")
            for source_batch_val, target_batch_val in progress_bar_val:
                source_batch_val, target_batch_val = source_batch_val.to(device), target_batch_val.to(device)
                t_val = torch.rand(source_batch_val.size(0), device=device) # Use random t for val loss consistency
                x_t_val = source_batch_val * (1-t_val.view(-1,1,1,1)) + target_batch_val * t_val.view(-1,1,1,1)
                
                pred_velocity_val = model(x_t_val, t_val)
                true_velocity_val = target_batch_val - source_batch_val
                val_loss_item = F.mse_loss(pred_velocity_val, true_velocity_val)
                epoch_val_loss += val_loss_item.item()
                progress_bar_val.set_postfix({"val_loss": val_loss_item.item()})
        
        avg_val_loss = epoch_val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        if val_batch_x_vis is not None:
            vis_path = os.path.join(vis_output_dir, f'epoch_{epoch+1}_efficient_rf.png')
            generate_visualization_efficient(model, ode_solver, val_batch_x_vis, val_batch_y_vis, vis_path)

        checkpoint_path = os.path.join(output_dir, f'checkpoint_efficient_epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss, 'val_loss': avg_val_loss,
        }, checkpoint_path)
        # print(f"Saved checkpoint to {checkpoint_path}")


        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model_efficient.pt'))
            # print(f"Saved best model to {os.path.join(output_dir, 'best_model_efficient.pt')}")
        else:
            patience_counter += 1
        
        if early_stop_patience is not None and patience_counter >= early_stop_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    if val_losses and not all(np.isinf(val_losses)): plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    plt.title("EfficientPretrainedFlow Training Loss")
    plt.savefig(os.path.join(output_dir, 'loss_curves_efficient_rf.png'))
    plt.close()
    # print(f"Saved loss curves to {os.path.join(output_dir, 'loss_curves_efficient_rf.png')}")
    
    print("Quick training for EfficientPretrainedFlow completed!")
    return model

def generate_visualization_efficient(model, ode_solver, source_batch, target_batch, output_path):
    model.eval()
    with torch.no_grad():
        trajectories = ode_solver.sample_ode(source_batch) # source_batch should be small (e.g. 4 samples)
        generated_t2 = trajectories[-1]
        generated_t2 = torch.clamp(generated_t2, 0, 1)

        n_samples = source_batch.size(0)
        fig, axes = plt.subplots(3, n_samples, figsize=(3 * n_samples, 9))
        if n_samples == 1: axes = axes.reshape(3,1)


        for i in range(n_samples):
            axes[0, i].imshow(source_batch[i, 0].cpu().numpy(), cmap='gray')
            axes[0, i].set_title('T1 Source'); axes[0, i].axis('off')
            axes[1, i].imshow(generated_t2[i, 0].cpu().numpy(), cmap='gray')
            axes[1, i].set_title('Generated T2'); axes[1, i].axis('off')
            axes[2, i].imshow(target_batch[i, 0].cpu().numpy(), cmap='gray')
            axes[2, i].set_title('True T2'); axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)
        # print(f"Saved efficient RF visualization to {output_path}")

        # Trajectory visualization
        if len(trajectories) > 2:
            traj_path = output_path.replace('.png', '_trajectory.png')
            fig_traj, axes_traj = plt.subplots(1, min(8, len(trajectories)), figsize=(20, 3)) # Adjusted figsize
            if min(8, len(trajectories)) == 1 : axes_traj = [axes_traj] # Make it iterable

            step_indices = np.linspace(0, len(trajectories) - 1, min(8, len(trajectories))).astype(int)
            for i, idx in enumerate(step_indices):
                axes_traj[i].imshow(trajectories[idx][0, 0].cpu().numpy(), cmap='gray') # Show 0-th sample in batch
                axes_traj[i].set_title(f'Step {idx}'); axes_traj[i].axis('off')
            plt.tight_layout()
            plt.savefig(traj_path)
            plt.close(fig_traj)
            # print(f"Saved efficient RF trajectory to {traj_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Efficient Pretrained Rectified Flow for MRI Translation')
    parser.add_argument('--t1_dir', type=str, required=True, help='Directory T1 slices')
    parser.add_argument('--t2_dir', type=str, required=True, help='Directory T2 slices')
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--test_ratio', type=float, default=0.15)
    parser.add_argument('--freeze_ratio', type=float, default=0.95)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--ode_steps', type=int, default=20, help="ODE steps for final eval & default vis")
    parser.add_argument('--ode_steps_vis', type=int, default=10, help="ODE steps for epoch vis (faster)")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=5) # More epochs for this one
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--output_dir', type=str, default='./results_efficient_rf')
    parser.add_argument('--early_stop', type=int, default=5, help="Early stopping patience")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.output_dir, f"run_efficient_{timestamp}")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Outputting to: {args.output_dir}")
    
    train_loader, val_loader, test_loader = create_data_loaders(
        args.t1_dir, args.t2_dir,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio,
        num_workers=0 if device.type == 'cpu' else 2
    )
    print(f"Loaded {len(train_loader.dataset)} training, {len(val_loader.dataset)} val, {len(test_loader.dataset)} test samples")
    
    model = EfficientPretrainedFlow(
        img_size=args.img_size, freeze_ratio=args.freeze_ratio, pretrained=True
    ).to(device)
    
    model = quick_train_efficient(
        model, train_loader, val_loader, device, args.output_dir,
        num_epochs=args.epochs, early_stop_patience=args.early_stop,
        img_size=args.img_size, ode_steps_vis=args.ode_steps_vis
    )
    
    print("EfficientPretrainedFlow training completed!")

    # Evaluate the model on the test set using evaluate_monai.py
    eval_output_dir = os.path.join(args.output_dir, "evaluation_efficient_rf")
    print(f"Evaluating model, final results in {eval_output_dir}")
    evaluate_model( # This function is from evaluate_monai.py
        model,      # The trained EfficientPretrainedFlow model
        test_loader,
        device,
        num_steps=args.ode_steps, # Number of steps for ODE solution during evaluation
        output_dir=os.path.join(eval_output_dir, f"eval_steps_{args.ode_steps}")
    )
    # Evaluate with 1 step as well, if reflow was intended or to see one-step quality
    evaluate_model(
        model,
        test_loader,
        device,
        num_steps=1,
        output_dir=os.path.join(eval_output_dir, "eval_steps_1")
    )