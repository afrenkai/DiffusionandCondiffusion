import torch
import torch.nn as nn
from consts import TIME_EMBED_DIM, T
from arch import UNet



if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNet(1, 1).to(device)
    model.load_state_dict(torch.load('diff_faces_unet.cpt'))
    # betas = torch.linspace(
    #     1e-4, 0.1, T, dtype = torch.float32
    # ).to(device)

