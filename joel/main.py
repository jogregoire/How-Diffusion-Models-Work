import torch
import argparse
import logging as log
from plot_images import *
from unet import *
from noise_scheduler import *
from sampler import *
from ddpm_sampler import *
from ddim_sampler import *
from gpuperf import *
from training import *

def main():
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--disable-cuda', action='store_true',help='Disable CUDA')
    parser.add_argument("--log", nargs='+', help="Provide logging level. Example --log debug'")
    parser.add_argument("--save-images", help="Save all images")
    parser.add_argument("--train", help="Train the model")
    args = parser.parse_args()

    # set log level
    log.basicConfig(level=args.log[0] if args.log else 'INFO')

    # run on GPU if available
    gpu_enabled = not args.disable_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if gpu_enabled else torch.device('cpu'))
    log.info('using device: %s', device)

    gpu_perf = GPUPerf(gpu_enabled, device)

    gpu_perf.snapshot('start')

    # network hyperparameters
    n_feat = 64 # 64 hidden dimension feature
    n_cfeat = 5 # context vector is of size 5
    height = 16 # 16x16 image
    in_channels=3 # rgb

    # load model
    model_filename = './weights/model_trained.pth'
    log.info(f'building model with in_channels={in_channels}, n_feat={n_feat}, n_cfeat={n_cfeat}, height={height}')
    nn_model = nn_model = ContextUnet(in_channels, n_feat=n_feat, n_cfeat=n_cfeat, height=height).to(device)
    
    # diffusion hyperparameters
    timesteps = 500

    # create noise scheduler
    noise = NoiseScheduler(timesteps, device, NoiseScheduler.LINEAR)
    sampler = DDPMSampler(noise)

    gpu_perf.snapshot('before')

    #args.train = True
    if args.train: # training ---------------------------
        # training hyperparameters
        batch_size = 100
        n_epoch = 32
        lrate=1e-3

        training = Training(noise, nn_model, lrate, batch_size, device=device, gpu_perf=gpu_perf)
        training.train(timesteps, n_epoch, model_filename, use_context=True)

        gpu_perf.save_snapshots(f'./data/gpu_snapshots_training_batch{batch_size}.csv')
    
    else: # sampling ------------------------------------

        # load model weights
        log.info(f'loading model {model_filename}')
        nn_model.load_state_dict(torch.load(model_filename, map_location=device))
        nn_model.eval()

        # sampling hyperparameters
        n_sample = 4

        # context
        # hero, non-hero, food, spell, side-facing
        #ctx = F.one_hot(torch.randint(0, 5, (32,)), 5).to(device=device).float()
        ctx = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]).to(device=device).float()

        # sample images
        samples = sampler.sample(n_sample=n_sample, height=height, timesteps=timesteps, nn_model=nn_model, device=device, gpu_perf=gpu_perf, context=ctx)

        gpu_perf.snapshot('after')

        # save generated images
        image_path = './data/'
        grid_filename = './data/grid.png'

        plot_grid(samples, grid_filename)

        if args.save_images:
            plot_images(samples, image_path)

        gpu_perf.save_snapshots('./data/gpu_snapshots_sampling.csv')

if __name__ == "__main__":
    main()