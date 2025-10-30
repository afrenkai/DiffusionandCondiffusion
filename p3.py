import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from consts import TIME_EMBED_DIM, T
from tqdm import trange

device = "cuda" if torch.cuda.is_available() else "cpu"

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )



def load_data(faces_path="faces23k_48x48.npy", ages_path="ages23k.npy"):
    print("Loading face and age data...")
    faces_data = np.load(faces_path)
    ages_data = np.load(ages_path)
    faces_data = faces_data / 255.0 * 2.0 - 1.0

    valid_indices = (ages_data >= 0) & (ages_data <= 100)
    faces_filtered = faces_data[valid_indices]
    ages_filtered = ages_data[valid_indices]

    faces_tensor = torch.tensor(faces_filtered, dtype=torch.float32).unsqueeze(1).to(device)
    ages_tensor = torch.tensor(ages_filtered, dtype=torch.float32).to(device)

    print(f"Filtered dataset size: {len(ages_tensor)}")
    return faces_tensor, ages_tensor

def sinusoidal_embedding(times):
    frequencies = torch.exp(
        torch.linspace(
            np.log(1.0),
            np.log(1000.),
            TIME_EMBED_DIM // 2
        )
    ).view(1, -1).to(times.device)
    angular_speeds = 2.0 * torch.pi * frequencies
    times = times.view(-1, 1).float()
    embeddings = torch.cat(
        [torch.sin(times.matmul(angular_speeds) / T), torch.cos(times.matmul(angular_speeds) / T)], dim=1
    )
    return embeddings

class AgeRegressor(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.dconv_down1 = double_conv(in_channels + TIME_EMBED_DIM, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)
        self.maxpool = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x, time_index):
        t_emb = sinusoidal_embedding(time_index)
        x = torch.cat(
            [x, t_emb.unsqueeze(-1).unsqueeze(-1).expand(x.size(0), -1, x.size(2), x.size(3))],
            dim=1
        )
        x = self.dconv_down1(x)
        x = self.maxpool(x)
        x = self.dconv_down2(x)
        x = self.maxpool(x)
        x = self.dconv_down3(x)
        x = self.maxpool(x)
        x = self.dconv_down4(x)
        x = self.global_pool(x).view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze()


class NoisedFaceDataset(Dataset):
    def __init__(self, faces, ages, alphas_bar, T, num_samples_per_image=5):
        self.faces = faces
        self.ages = ages
        self.alphas_bar = alphas_bar
        self.T = T
        self.num_samples_per_image = num_samples_per_image

    def __len__(self):
        return len(self.faces) * self.num_samples_per_image

    def __getitem__(self, idx):
        img_idx = idx // self.num_samples_per_image
        x0 = self.faces[img_idx]
        age = self.ages[img_idx]

        t = torch.randint(0, self.T, (1,)).item()
        sqrt_alpha_bar = torch.sqrt(self.alphas_bar[t])
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alphas_bar[t])
        noise = torch.randn_like(x0)
        zt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
        return zt, t, age


def train(model, train_loader, optimizer, criterion, num_epochs=10):
    model.train()
    for epoch in trange(num_epochs):
        total_loss = 0
        for zt, t, age in train_loader:
            zt, t, age = zt.to(device), t.to(device), age.to(device)
            optimizer.zero_grad()
            age_pred = model(zt, t)
            loss = criterion(age_pred, age)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")
    print("Training complete!")


def ddpm_sample_with_guidance(model, age_regressor, target_age, num_samples=10, img_size=48, guidance_scale=0.2):
    model.eval()
    age_regressor.eval()
    x = torch.randn(num_samples, 1, img_size, img_size).to(device)

    for t in reversed(range(T)):
        t_tensor = torch.tensor([t] * num_samples).to(device)
        x.requires_grad_(True)

        predicted_noise = model(x, t_tensor)
        age_pred = age_regressor(x, t_tensor)
        loss = ((age_pred - target_age) ** 2).sum()
        grad = torch.autograd.grad(loss, x)[0]

        x = x.detach()
        guided_noise = predicted_noise - guidance_scale * grad

        alpha_t = alphas[t]
        alpha_bar_t = alphas_bar[t]
        beta_t = betas[t]
        mean = (1.0 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * guided_noise)

        if t > 0:
            z = torch.randn_like(x)
            sigma_t = torch.sqrt(beta_t)
            x = mean + sigma_t * z
        else:
            x = mean

    return x


def visualize_faces(faces_tensor, model, target_ages, num_samples_per_age=10):
    all_generated_faces = []
    for target_age in target_ages:
        print(f"Generating {num_samples_per_age} faces for age {target_age}...")
        generated = ddpm_sample_with_guidance(
            model,
            age_regressor,
            target_age=torch.tensor([target_age]*num_samples_per_age).to(device),
            num_samples=num_samples_per_age,
            img_size=48,
            guidance_scale=0.2
        )
        all_generated_faces.append(generated)

    all_generated_faces = torch.cat(all_generated_faces, dim=0).cpu().numpy()
    all_generated_faces = (all_generated_faces + 1.0) / 2.0
    all_generated_faces = np.clip(all_generated_faces, 0, 1)

    fig, axes = plt.subplots(len(target_ages), num_samples_per_age, figsize=(20, 8))
    fig.suptitle("Age-Conditioned Face Generation using Classifier Guidance", fontsize=18, weight="bold")
    for row, target_age in enumerate(target_ages):
        for col in range(num_samples_per_age):
            idx = row * num_samples_per_age + col
            axes[row, col].imshow(all_generated_faces[idx, 0], cmap="gray")
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(f"Age {target_age}", fontsize=14, weight="bold", rotation=0, labelpad=40)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    faces_tensor, ages_tensor = load_data()
    age_regressor = AgeRegressor().to(device)
    print(f"Age regressor parameters: {sum(p.numel() for p in age_regressor.parameters()):,}")

    betas = torch.linspace(1e-4, 0.01, T, dtype=torch.float32).to(device)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)


    train_dataset = NoisedFaceDataset(faces_tensor, ages_tensor, alphas_bar, T, num_samples_per_image=5)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)

    optimizer = torch.optim.Adam(age_regressor.parameters(), lr=1e-4)
    criterion = nn.MSELoss()


    train(age_regressor, train_loader, optimizer, criterion, num_epochs=10)

    target_ages = [18, 40, 60, 80]
    visualize_faces(faces_tensor, model  =age_regressor, target_ages = target_ages, num_samples_per_age=10)
