import torch
import argparse
import logging as log
from plot_images import *
from model import *
from noise_scheduler import *
from sampler import *
from ddpm_sampler import *

def main():
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--disable-cuda', action='store_true',help='Disable CUDA')
    parser.add_argument("-log", "--log", nargs='+', help="Provide logging level. Example --log debug'")
    args = parser.parse_args()

    # set log level
    logging.basicConfig(level=args.log[0] if args.log else 'INFO')

    # run on GPU if available
    gpu_enabled = not args.disable_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if gpu_enabled else torch.device('cpu'))
    log.info('using device: %s', device)

    # network hyperparameters
    n_feat = 64 # 64 hidden dimension feature
    n_cfeat = 5 # context vector is of size 5
    height = 16 # 16x16 image

    # load model
    model_path = './weights/'
    nn_model = Model(in_channels=3, n_feat=n_feat, n_cfeat=n_cfeat, height=height, device=device)
    nn_model.load_model(model_path)

    # diffusion hyperparameters
    timesteps = 500
    n_sample = 4

    # sample images
    noise = NoiseScheduler(timesteps, device, NoiseScheduler.LINEAR)
    sampler = DDPMSampler(noise)
    samples = sampler.sample(n_sample=n_sample, height=height, timesteps=timesteps, nn_model=nn_model, device=device)

    # save generated images
    image_path = './data/'
    grid_filename = './data/grid.png'

    plot_grid(samples, grid_filename)
    plot_images(samples, image_path)

if __name__ == "__main__":
    main()