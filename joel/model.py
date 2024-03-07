import logging
from unet import *

class Model():
    def __init__(self, in_channels, n_feat, n_cfeat, height, device):
        logging.info(f'building model with in_channels={in_channels}, n_feat={n_feat}, n_cfeat={n_cfeat}, height={height}')
        self.nn_model = nn_model = ContextUnet(in_channels=3, n_feat=n_feat, n_cfeat=n_cfeat, height=height).to(device)
        self.device = device

    def load_model(self, path):
        logging.info(f'loading model from {path}')
        self.nn_model.load_state_dict(torch.load(f"{path}/model_trained.pth", map_location=self.device))
        self.nn_model.eval()

    def predict(self, x, timestep):
        # predict noise e_(x_t,t)
        return self.nn_model(x, timestep)
