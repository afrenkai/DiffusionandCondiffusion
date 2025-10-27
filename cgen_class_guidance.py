import torch
import torch.nn as nn
from consts import T, TIME_EMBED_DIM
from arch import UNet, sinusoidal_positional_encoding, double_conv
from ds import FacesDataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class AgeRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.dconv_down1 = double_conv(1 + TIME_EMBED_DIM, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)
        self.maxpool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(512 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x, time_index):
        time_embedding = sinusoidal_positional_encoding(time_index)
        x = torch.cat([x, time_embedding.unsqueeze(-1).unsqueeze(-1).expand(x.size(0), -1, x.size(2), x.size(3))], dim=1)
        x = self.dconv_down1(x)
        x = self.maxpool(x)
        x = self.dconv_down2(x)
        x = self.maxpool(x)
        x = self.dconv_down3(x)
        x = self.maxpool(x)
        x = self.dconv_down4(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(-1)

def diffusion_kernel(x, t, betas, alphas_prod, device):
    alpha_bar_t = alphas_prod[t]
    sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_t)
    noise = torch.randn_like(x).to(device)
    z_t = sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise
    return z_t

def train_regressor():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    faces = np.load('faces23k_48x48.npy')
    ages = np.load('ages23k.npy')
    
    valid_mask = (ages >= 0) & (ages <= 100)
    faces = faces[valid_mask]
    ages = ages[valid_mask]
    
    faces = faces / 255.0
    faces = (faces - 0.5) / 0.5
    
    regressor = AgeRegressor().to(device)
    optimizer = torch.optim.Adam(regressor.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    betas = torch.linspace(1e-4, 0.1, T, dtype=torch.float32).to(device)
    alphas = 1.0 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    
    num_epochs = 20
    batch_size = 64
    
    for epoch in range(num_epochs):
        regressor.train()
        total_loss = 0
        num_batches = 0
        
        indices = np.random.permutation(len(faces))
        
        for i in tqdm(range(0, len(faces), batch_size), desc=f'Epoch {epoch+1}/{num_epochs}'):
            batch_indices = indices[i:i+batch_size]
            batch_faces = torch.from_numpy(faces[batch_indices]).unsqueeze(1).float().to(device)
            batch_ages = torch.from_numpy(ages[batch_indices]).float().to(device)
            
            t_vals = torch.randint(0, T, (len(batch_faces),), device=device)
            
            z_t_list = []
            for j, t in enumerate(t_vals):
                z_t = diffusion_kernel(batch_faces[j:j+1], t.item(), betas, alphas_prod, device)
                z_t_list.append(z_t)
            z_t = torch.cat(z_t_list, dim=0)
            
            optimizer.zero_grad()
            pred_ages = regressor(z_t, t_vals)
            loss = criterion(pred_ages, batch_ages)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')
    
    torch.save(regressor.state_dict(), 'age_regressor.pth')
    print("Saved age_regressor.pth")
    
    return regressor

def sample_with_guidance(diffusion_model, regressor, target_age, num_samples, guidance_scale=0.2):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    betas = torch.linspace(1e-4, 0.1, T, dtype=torch.float32).to(device)
    alphas = 1.0 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    sqrt_one_minus_alphas_prod = torch.sqrt(1.0 - alphas_prod)
    
    z_t = torch.randn(num_samples, 1, 48, 48).to(device)
    
    diffusion_model.eval()
    regressor.eval()
    
    for t_idx in tqdm(reversed(range(T)), desc=f'Generating age {target_age}'):
        t_batch = torch.full((num_samples,), t_idx, device=device, dtype=torch.long)
        
        z_t.requires_grad_(True)
        
        with torch.enable_grad():
            pred_age = regressor(z_t, t_batch)
            age_loss = ((pred_age - target_age) ** 2).sum()
            grad = torch.autograd.grad(age_loss, z_t)[0]
        
        z_t.requires_grad_(False)
        
        with torch.no_grad():
            predicted_noise = diffusion_model(z_t, t_batch)
            
            predicted_noise = predicted_noise - guidance_scale * grad
            
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
    
    regressor = train_regressor()
    
    diffusion_model = UNet(1, 1).to(device)
    diffusion_model.load_state_dict(torch.load('diff_unet_faces.cpt', map_location=device))
    diffusion_model.eval()
    
    regressor.load_state_dict(torch.load('age_regressor.pth', map_location=device))
    
    target_ages = [18, 40, 60, 80]
    
    fig, axes = plt.subplots(4, 10, figsize=(15, 6))
    
    for row, age in enumerate(target_ages):
        samples = sample_with_guidance(diffusion_model, regressor, age, 10)
        samples = samples.cpu().numpy()
        samples = (samples + 1.0) / 2.0
        samples = np.clip(samples, 0, 1)
        
        for col in range(10):
            axes[row, col].imshow(samples[col, 0], cmap='gray')
            axes[row, col].axis('off')
        
        axes[row, 0].set_ylabel(f'Age {age}', rotation=0, size='large', labelpad=40)
    
    plt.tight_layout()
    plt.savefig('conditional_generation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved conditional_generation.png")
