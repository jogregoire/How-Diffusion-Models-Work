import torch

class NoiseScheduler():
    LINEAR = 0
    COSINE = 1

    def __init__(self, timesteps, device, shape = LINEAR, beta1 = 1e-4, beta2 = 0.02):
        if shape == self.LINEAR:
            self.linear_schedule(timesteps, device, beta1, beta2)
        elif shape == self.COSINE:
            raise NotImplementedError
        else:
            raise NotImplementedError

    def linear_schedule(self, timesteps, device, beta1 = 1e-4, beta2 = 0.02):
        # hyperparameters
        self.beta1 = beta1
        self.beta2 = beta2

        # construct linear noise schedule
        self.b_t = (self.beta2 - self.beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + self.beta1
        self.a_t = 1 - self.b_t
        self.ab_t = torch.cumsum(self.a_t.log(), dim=0).exp()    
        self.ab_t[0] = 1
        self.bsqr_t = self.b_t.sqrt()

    # helper function for sampling; removes the predicted noise (but adds some noise back in to avoid collapse)
    def denoise_add_noise(self, x, t, pred_noise, z=None):
        if z is None:
            z = torch.randn_like(x)
        noise = self.b_t.sqrt()[t] * z
        mean = (x - pred_noise * ((1 - self.a_t[t]) / (1 - self.ab_t[t]).sqrt())) / self.a_t[t].sqrt()
        return mean + noise
    
    # helper function for training: perturbs an image to a specified noise level
    def perturb_input(self, x, t, noise):
        return self.ab_t.sqrt()[t, None, None, None] * x + (1 - self.ab_t[t, None, None, None]) * noise