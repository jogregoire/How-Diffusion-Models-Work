import torch
import logging as log
from sampler import Sampler

class DDIMSampler(Sampler):
    def __init__(self, noise_sampler):
        self.noise_sampler = noise_sampler

    def denoise_ddim(self, x, t, t_prev, pred_noise):
        ab = self.noise_sampler.ab_t[t]
        ab_prev = self.noise_sampler.ab_t[t_prev]
        
        x0_pred = ab_prev.sqrt() / ab.sqrt() * (x - (1 - ab).sqrt() * pred_noise)
        dir_xt = (1 - ab_prev).sqrt() * pred_noise

        return x0_pred + dir_xt
    
    @torch.no_grad()
    def sample(self, n_sample, height, timesteps, nn_model, device, gpu_perf, context=None, stop_after_timesteps=None):
        
        end = 0
        if stop_after_timesteps is not None:
            end = timesteps - stop_after_timesteps
            log.info(f"end: {end}")

        n=20

        # x_T ~ N(0, 1), sample initial noise
        samples = torch.randn(n_sample, 3, height, height).to(device)  

        step_size = timesteps // n

        # array to keep track of generated steps for plotting
        for i in range(timesteps, end, -step_size):
            print(f'sampling timestep {i:3d}\r', end='\r') # not a log

            # reshape time tensor
            t = torch.tensor([i / timesteps])[:, None, None, None].to(device)

            # predict noise e_(x_t,t)
            if context is not None:
                eps = nn_model(samples, t, c=context)
            else:
                eps = nn_model(samples, t)

            samples = self.denoise_ddim(samples, i, i - step_size, eps)

            gpu_perf.snapshot(f"timestep {i:3d}")

        return samples