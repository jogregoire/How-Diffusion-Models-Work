import torch
import logging as log
from sampler import Sampler

class DDPMSampler(Sampler):
    def __init__(self, noise_sampler):
        self.noise_sampler = noise_sampler

    # helper function for sampling; removes the predicted noise (but adds some noise back in to avoid collapse)
    def denoise_add_noise(self, x, t, pred_noise, z=None):
        s = self.noise_sampler
        if z is None:
            z = torch.randn_like(x)
        noise = s.b_t.sqrt()[t] * z
        mean = (x - pred_noise * ((1 - s.a_t[t]) / (1 - s.ab_t[t]).sqrt())) / s.a_t[t].sqrt()
        return mean + noise

    @torch.no_grad()
    def sample(self, n_sample, height, timesteps, nn_model, device, gpu_perf, context=None, stop_after_timesteps=None):
        # x_T ~ N(0, 1), sample initial noise
        samples = torch.randn(n_sample, 3, height, height).to(device) 

        end = 0
        if stop_after_timesteps is not None:
            end = timesteps - stop_after_timesteps
            log.info(f"end: {end}")

        # array to keep track of generated steps for plotting
        for i in range(timesteps, end, -1):
            print(f'sampling timestep {i:3d}\r', end='\r') # not a log

            # reshape time tensor
            t = torch.tensor([i / timesteps])[:, None, None, None].to(device)

            # sample some random noise to inject back in. For i = 1, don't add back in noise
            z = torch.randn_like(samples) if i > 1 else 0

            # predict noise e_(x_t,t)
            if context is not None:
                eps = nn_model(samples, t, c=context)
            else:
                eps = nn_model(samples, t)

            samples = self.denoise_add_noise(samples, i, eps, z)

            gpu_perf.snapshot(f"timestep {i:3d}")

        return samples