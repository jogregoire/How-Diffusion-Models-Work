import torch
from sampler import Sampler
from model import *

class DDPMSampler(Sampler):
    def __init__(self, noise_sampler):
        self.noise_sampler = noise_sampler

    @torch.no_grad()
    def sample(self, n_sample, height, timesteps, nn_model, device):
        # x_T ~ N(0, 1), sample initial noise
        samples = torch.randn(n_sample, 3, height, height).to(device)  

        # array to keep track of generated steps for plotting
        for i in range(timesteps, 0, -1):
            print(f'sampling timestep {i:3d}', end='\r')

            # reshape time tensor
            t = torch.tensor([i / timesteps])[:, None, None, None].to(device)

            # sample some random noise to inject back in. For i = 1, don't add back in noise
            z = torch.randn_like(samples) if i > 1 else 0

            eps = nn_model.predict(samples, t)
            samples = self.noise_sampler.denoise_add_noise(samples, i, eps, z)

        return samples