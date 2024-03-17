import torch
import logging as log
from sampler import Sampler

class DDPMSampler(Sampler):
    def __init__(self, noise_sampler):
        self.noise_sampler = noise_sampler

    @torch.no_grad()
    def sample(self, n_sample, height, timesteps, nn_model, device, gpu_perf):
        # x_T ~ N(0, 1), sample initial noise
        samples = torch.randn(n_sample, 3, height, height).to(device)  

        # array to keep track of generated steps for plotting
        for i in range(timesteps, 0, -1):
            print(f'sampling timestep {i:3d}\r', end='\r') # not a log

            # reshape time tensor
            t = torch.tensor([i / timesteps])[:, None, None, None].to(device)

            # sample some random noise to inject back in. For i = 1, don't add back in noise
            z = torch.randn_like(samples) if i > 1 else 0

            # predict noise e_(x_t,t)
            eps = nn_model(samples, t)
            samples = self.noise_sampler.denoise_add_noise(samples, i, eps, z)

            gpu_perf.snapshot(f"timestep {i:3d}")

        return samples