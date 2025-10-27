import torch
import torch.nn as nn
from consts import T
from arch import UNet
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def sample(model: nn.Module, batch_size: int, channels: int, height: int, width: int):
    model.eval()
    device = next(model.parameters()).device
    
    betas = torch.linspace(1e-4, 0.1, T, dtype=torch.float32).to(device)
    alphas = 1.0 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    sqrt_one_minus_alphas_prod = torch.sqrt(1.0 - alphas_prod)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    
    z_t = torch.randn(batch_size, channels, height, width).to(device)
    
    with torch.no_grad():
        for t_idx in tqdm(reversed(range(T)), desc='Sampling'):
            t_batch = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)
            
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
                
                # Posterior mean formula
                coef1 = beta_t * torch.sqrt(alpha_bar_t_prev) / (1.0 - alpha_bar_t)
                coef2 = (1.0 - alpha_bar_t_prev) * torch.sqrt(alpha_t) / (1.0 - alpha_bar_t)
                
                posterior_mean = coef1 * pred_x0 + coef2 * z_t
                
                # Posterior variance
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
    
    samples = sample(model, 100, 1, 48, 48)
    samples = samples.cpu().numpy()
    samples = (samples + 1.0) / 2.0
    samples = np.clip(samples, 0, 1)
    
    fig, axes = plt.subplots(10, 10, figsize=(15, 15))
    for i in range(10):
        for j in range(10):
            idx = i * 10 + j
            axes[i, j].imshow(samples[idx, 0], cmap='gray')
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.savefig('unconditional_samples.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved unconditional_samples.png")
