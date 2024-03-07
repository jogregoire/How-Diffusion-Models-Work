from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.utils import save_image, make_grid
import numpy as np
from diffusion_utilities import *
from unet import *
from PIL import Image


class Noise:
    def __init__(self, timesteps, device):
        # diffusion hyperparameters    
        self.beta1 = 1e-4
        self.beta2 = 0.02

        # construct DDPM noise schedule
        self.b_t = (self.beta2 - self.beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + self.beta1
        self.a_t = 1 - self.b_t
        self.ab_t = torch.cumsum(self.a_t.log(), dim=0).exp()    
        self.ab_t[0] = 1

    # helper function; removes the predicted noise (but adds some noise back in to avoid collapse)
    def denoise_add_noise(self, x, t, pred_noise, z=None):
        if z is None:
            z = torch.randn_like(x)
        noise = self.b_t.sqrt()[t] * z
        mean = (x - pred_noise * ((1 - self.a_t[t]) / (1 - self.ab_t[t]).sqrt())) / self.a_t[t].sqrt()
        return mean + noise

# sample using standard algorithm
@torch.no_grad()
def sample_ddpm(n_sample, height, timesteps, noise, nn_model, device):
    # x_T ~ N(0, 1), sample initial noise
    samples = torch.randn(n_sample, 3, height, height).to(device)  

    # array to keep track of generated steps for plotting
    for i in range(timesteps, 0, -1):
        print(f'sampling timestep {i:3d}', end='\r')

        # reshape time tensor
        t = torch.tensor([i / timesteps])[:, None, None, None].to(device)

        # sample some random noise to inject back in. For i = 1, don't add back in noise
        z = torch.randn_like(samples) if i > 1 else 0

        eps = nn_model(samples, t)    # predict noise e_(x_t,t)
        samples = noise.denoise_add_noise(samples, i, eps, z)

    return samples

def main():
    # diffusion hyperparameters
    timesteps = 500

    # network hyperparameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
    n_feat = 64 # 64 hidden dimension feature
    n_cfeat = 5 # context vector is of size 5
    height = 16 # 16x16 image
    save_dir = './weights/'

    # construct model
    print("construct model")
    nn_model = ContextUnet(in_channels=3, n_feat=n_feat, n_cfeat=n_cfeat, height=height).to(device)

    # load in model weights and set to eval mode
    nn_model.load_state_dict(torch.load(f"{save_dir}/model_trained.pth", map_location=device))
    nn_model.eval()
    print("Loaded in Model")

    noise = Noise(timesteps, device)

    n_sample = 2

    samples = sample_ddpm(n_sample=n_sample, height=height, timesteps=timesteps, noise=noise, nn_model=nn_model, device=device)

    x = samples.detach().cpu().numpy()

    for i in range(2):
        pil_image = np.transpose(x[i], (1, 2, 0)) # -1.5 to 1.5
        pil_image = unorm(pil_image) # 0..1
        pil_image = (pil_image*255).astype(np.uint8) # 0..255
        im = Image.fromarray(pil_image)
        im.save(f'out{i}.png')

if __name__ == "__main__":
    main()