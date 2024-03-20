import torch

class NoiseScheduler():
    LINEAR = 0
    QUADRATIC = 1
    SIGMOID = 2
    COSINE = 3

    def __init__(self, timesteps, device, shape = LINEAR, beta1 = 1e-4, beta2 = 0.02):
        if shape == self.LINEAR:
            self.linear_schedule(timesteps, device, beta1, beta2)
        elif shape == self.QUADRATIC:
            self.quadratic_schedule(timesteps, device, beta1, beta2)
        elif shape == self.SIGMOID:
            self.sigmoid_schedule(timesteps, device, beta1, beta2)
        elif shape == self.COSINE:
            self.cosine_schedule(timesteps, device, beta1, beta2)
        else:
            raise NotImplementedError

    def linear_schedule(self, timesteps, device, beta1 = 1e-4, beta2 = 0.02):
        # construct linear noise schedule
        self.b_t = torch.linspace(beta1, beta2, timesteps, device=device)
        self.__define_alphas()

    def quadratic_schedule(self, timesteps, device, beta1 = 1e-4, beta2 = 0.02):
        # construct quadratic noise schedule
        self.b_t = torch.linspace(beta1**0.5, beta2**0.5, timesteps, device=device) ** 2
        self.__define_alphas()

    def sigmoid_schedule(self, timesteps, device, beta1 = 1e-4, beta2 = 0.02):
        # proposed in https://arxiv.org/abs/2212.11972 - Figure 8
        # better for images > 64x64, when used during training
        betas = torch.linspace(-6, 6, timesteps, device=device)
        self.b_t = torch.sigmoid(betas) * (beta2 - beta1) + beta1
        self.__define_alphas()

    def cosine_schedule(self, timesteps, device, beta1 = 1e-4, beta2 = 0.02):
        # cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        s=0.008
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, device=device)
        self.ab_t = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        self.ab_t = self.ab_t / self.ab_t[0]
        betas = 1 - (self.ab_t[1:] / self.ab_t[:-1])
        self.b_t = torch.clip(betas, 0.0001, 0.9999)
        self.a_t = 1 - self.b_t
        self.bsqr_t = self.b_t.sqrt()
        
    def __define_alphas(self):
        self.a_t = 1 - self.b_t
        self.ab_t = torch.cumsum(self.a_t.log(), dim=0).exp()    
        self.ab_t[0] = 1
        self.bsqr_t = self.b_t.sqrt()

    # helper function for training: perturbs an image to a specified noise level
    def perturb_input(self, x, t, noise):
        return self.ab_t.sqrt()[t, None, None, None] * x + (1 - self.ab_t[t, None, None, None]) * noise
    
