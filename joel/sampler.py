from abc import ABC, abstractmethod 

class Sampler(ABC):
    @abstractmethod
    def sample(self, n_sample, height, timesteps, nn_model, device):
        pass