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
    parser.add_argument("--log", nargs='+', help="Provide logging level. Example --log debug'")
    parser.add_argument("--train", help="Train the model")
    parser.add_argument("--timesteps", help="Timesteps")
    args = parser.parse_args()

    device, gpu_perf = initialize(args.log[0] if args.log else 'INFO')

    # diffusion hyperparameters
    if args.timesteps:
        timesteps = int(args.timesteps)
    else:
        timesteps = 500

    noise_type = NoiseScheduler.LINEAR

    args.train = True
    if args.train:
        # training hyperparameters
        batch_size = 100
        n_epoch = 32
        use_context=True
        train(device, gpu_perf, timesteps, noise_type, batch_size, n_epoch, use_context)
    else:
        # sampling hyperparameters
        n_sample = 9
        sample(device, gpu_perf, timesteps, noise_type, n_sample)

def initialize(logLevel = None):
    # set log level
    log.basicConfig(level=logLevel if logLevel else 'INFO')

    # run on GPU if available
    gpu_enabled = torch.cuda.is_available()
    device = torch.device("cuda:0" if gpu_enabled else torch.device('cpu'))
    log.info('using device: %s', device)

    gpu_perf = GPUPerf(gpu_enabled, device)

    gpu_perf.snapshot('start')

    return device, gpu_perf

def train(device, gpu_perf, timesteps, noise_type, batch_size, n_epoch, use_context): 
        # network hyperparameters
        n_feat = 64 # 64 hidden dimension feature
        n_cfeat = 5 # context vector is of size 5
        height = 16 # 16x16 image
        in_channels=3 # rgb

        lrate=1e-3

        log.info(f"training:: timesteps:{timesteps}, noise:{noise_type}, batch size:{batch_size}, epoch: {n_epoch}")

        # load model
        model_filename = f"./weights/model_trained_{timesteps}_{noise_type}_{batch_size}_{n_epoch}_context.pth"

        log.info(f'building model with in_channels={in_channels}, n_feat={n_feat}, n_cfeat={n_cfeat}, height={height}')
        nn_model = nn_model = ContextUnet(in_channels, n_feat=n_feat, n_cfeat=n_cfeat, height=height).to(device)
        
        # create noise scheduler
        noise = NoiseScheduler(timesteps, device, noise_type)
        sampler = DDPMSampler(noise)

        gpu_perf.snapshot('model_loaded')

        context_value = "no_context" if use_context == False else "context"
        model_filename = f"./weights/model_trained_{timesteps}_{noise_type}_{batch_size}_{n_epoch}_{context_value}.pth"

        training = Training(noise, nn_model, lrate, batch_size, device=device, gpu_perf=gpu_perf)
        training.train(timesteps, n_epoch, model_filename, use_context)

        gpu_perf.save_snapshots(f'./data/gpu_snapshots_training_batch{batch_size}.csv')

def sample(device, gpu_perf, timesteps, noise_type, n_sample, sampler_type = 'DDPM', model_filename = None, context = None, grid_filename = './data/grid.png'):
        # network hyperparameters
        n_feat = 64 # 64 hidden dimension feature
        n_cfeat = 5 # context vector is of size 5
        height = 16 # 16x16 image
        in_channels=3 # rgb

        log.info(f"sampling:: timesteps: {timesteps}, noise: {noise_type}, n_sample: {n_sample}, sampler: {sampler_type}")

        # load model
        if model_filename == None:
            model_filename = f"./weights/model_trained_500_{noise_type}_100_32_context.pth"

        log.info(f'building model with in_channels={in_channels}, n_feat={n_feat}, n_cfeat={n_cfeat}, height={height}')
        nn_model = nn_model = ContextUnet(in_channels, n_feat=n_feat, n_cfeat=n_cfeat, height=height).to(device)
        
        # create noise scheduler
        noise = NoiseScheduler(timesteps, device, noise_type)
        if sampler_type == 'DDPM':
            sampler = DDPMSampler(noise)
        else:
            sampler = DDIMSampler(noise)

        gpu_perf.snapshot('model_loaded')

        # load model weights
        log.info(f'loading model {model_filename}')
        nn_model.load_state_dict(torch.load(model_filename, map_location=device))
        nn_model.eval()

        # context
        # hero, non-hero, food, spell, side-facing
        #ctx = F.one_hot(torch.randint(0, 5, (32,)), 5).to(device=device).float()
        ctx = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]).to(device=device).float()
        if context != None:
             ctx = torch.tensor(context).to(device=device).float()

        # sample images
        samples = sampler.sample(n_sample=n_sample, height=height, timesteps=timesteps, nn_model=nn_model, device=device, gpu_perf=gpu_perf, context=ctx)

        gpu_perf.snapshot('after')

        # save generated images
        image_path = './data/'

        plot_grid(samples, grid_filename)
        # plot_images(samples, image_path) # save individual images

        gpu_perf.save_snapshots('./data/gpu_snapshots_sampling.csv')

if __name__ == "__main__":
    main()