import torch
import torch.nn as nn
from consts import T
from arch import UNet
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def diffusion_kernel(x, t, betas, alphas_prod, device):
    alpha_bar_t = alphas_prod[t]
    sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_t)
    noise = torch.randn_like(x).to(device)
    z_t = sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise
    return z_t

def sample_from_zt(model, z_t, t_start, device):
    betas = torch.linspace(1e-4, 0.1, T, dtype=torch.float32).to(device)
    alphas = 1.0 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    sqrt_one_minus_alphas_prod = torch.sqrt(1.0 - alphas_prod)
    
    with torch.no_grad():
        for t_idx in range(t_start - 1, -1, -1):
            t_batch = torch.full((z_t.size(0),), t_idx, device=device, dtype=torch.long)
            
            predicted_noise = model(z_t, t_batch)
            
            beta_t = betas[t_idx]
            alpha_t = alphas[t_idx]
            
            # Predict x_0 from z_t and noise
            pred_x0 = (z_t - sqrt_one_minus_alphas_prod[t_idx] * predicted_noise) / torch.sqrt(alphas_prod[t_idx])
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
            
            if t_idx > 0:
                # Compute posterior mean
                alpha_bar_t = alphas_prod[t_idx]
                alpha_bar_t_prev = alphas_prod[t_idx - 1]
                
                coef1 = beta_t * torch.sqrt(alpha_bar_t_prev) / (1.0 - alpha_bar_t)
                coef2 = (1.0 - alpha_bar_t_prev) * torch.sqrt(alpha_t) / (1.0 - alpha_bar_t)
                
                posterior_mean = coef1 * pred_x0 + coef2 * z_t
                
                posterior_variance = beta_t * (1.0 - alpha_bar_t_prev) / (1.0 - alpha_bar_t)
                
                noise = torch.randn_like(z_t)
                z_t = posterior_mean + torch.sqrt(posterior_variance) * noise
            else:
                z_t = pred_x0
    
    return z_t

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNet(1, 1).to(device)
    model.load_state_dict(torch.load('diff_unet_faces.cpt', map_location=device))
    model.eval()
    
    faces = np.load('faces23k_48x48.npy')
    faces = faces / 255.0
    faces = (faces - 0.5) / 0.5
    
    betas = torch.linspace(1e-4, 0.1, T, dtype=torch.float32).to(device)
    alphas = 1.0 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    
    t_values = [100, 300, 500, 700]
    num_pairs = 5
    
    fig, axes = plt.subplots(4, 15, figsize=(22, 6))
    
    for row, t in enumerate(t_values):
        for pair_idx in range(num_pairs):
            idx1 = np.random.randint(0, len(faces))
            idx2 = np.random.randint(0, len(faces))
            
            x1 = torch.from_numpy(faces[idx1]).unsqueeze(0).unsqueeze(0).float().to(device)
            x2 = torch.from_numpy(faces[idx2]).unsqueeze(0).unsqueeze(0).float().to(device)
            
            z1_t = diffusion_kernel(x1, t, betas, alphas_prod, device)
            z2_t = diffusion_kernel(x2, t, betas, alphas_prod, device)
            
            z_t_merged = torch.zeros_like(z1_t)
            z_t_merged[:, :, :, :24] = z1_t[:, :, :, :24]
            z_t_merged[:, :, :, 24:] = z2_t[:, :, :, 24:]
            
            merged_sample = sample_from_zt(model, z_t_merged, t, device)
            
            x1_img = x1.cpu().numpy()[0, 0]
            x2_img = x2.cpu().numpy()[0, 0]
            merged_img = merged_sample.cpu().numpy()[0, 0]
            
            x1_img = (x1_img + 1.0) / 2.0
            x2_img = (x2_img + 1.0) / 2.0
            merged_img = (merged_img + 1.0) / 2.0
            
            x1_img = np.clip(x1_img, 0, 1)
            x2_img = np.clip(x2_img, 0, 1)
            merged_img = np.clip(merged_img, 0, 1)
            
            axes[row, pair_idx * 3].imshow(x1_img, cmap='gray')
            axes[row, pair_idx * 3].axis('off')
            axes[row, pair_idx * 3 + 1].imshow(x2_img, cmap='gray')
            axes[row, pair_idx * 3 + 1].axis('off')
            axes[row, pair_idx * 3 + 2].imshow(merged_img, cmap='gray')
            axes[row, pair_idx * 3 + 2].axis('off')
            
            if pair_idx == 0:
                axes[row, 0].set_ylabel(f't={t}', rotation=0, size='large', labelpad=30)
    
    plt.tight_layout()
    plt.savefig('inpainting_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved inpainting_results.png")
