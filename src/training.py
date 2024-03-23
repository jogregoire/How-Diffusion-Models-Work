import torch
import logging as log
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from gpuperf import *
from dataset import *
from tqdm import tqdm
    
class Training():
    def __init__(self, noise_sampler, nn_model, lr, batch_size, device, gpu_perf):
        self.noise_sampler = noise_sampler
        self.nn_model = nn_model
        self.lr = lr
        self.device = device
        self.gpu_perf = gpu_perf

        # load dataset and construct optimizer
        dataset = CustomDataset("./dataset/sprites_1788_16x16.npy", "./dataset/sprite_labels_nc_1788_16x16.npy", null_context=False)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        # https://arxiv.org/pdf/1412.6980.pdf
        # The paper "Adam: A Method for Stochastic Optimization" recommend using a learning rate of 0.001,
        # Extension gradient descent that automatically adapts a learning rate for each input variable
        # https://machinelearningmastery.com/adam-optimization-from-scratch/
        self.optim = torch.optim.Adam(nn_model.parameters(), lr=lr)

    def train(self, timesteps, n_epoch, filename, use_context=False):
        # training without context code
        # set into train mode
        self.nn_model.train()

        for ep in range(n_epoch):
            log.info(f'epoch {ep}')
            
            # linearly decay learning rate
            self.optim.param_groups[0]['lr'] = self.lr*(1-ep/n_epoch)
            
            pbar = tqdm(self.dataloader, mininterval=2 )
            for x, c in pbar:   # x: images
                self.optim.zero_grad()
                x = x.to(self.device)

                if use_context:
                    #----------------- context code -----------------
                    c = c.to(x)
                    #----------------- context code -----------------
                    c = c.to(x) # move c to same device as x
            
                    # randomly mask out c
                    context_mask = torch.bernoulli(torch.zeros(c.shape[0]) + 0.9).to(self.device)
                    c = c * context_mask.unsqueeze(-1)
                
                # perturb data
                noise = torch.randn_like(x)
                t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(self.device) 
                x_pert = self.noise_sampler.perturb_input(x, t, noise)
                

                # use network to recover noise
                if use_context:
                    pred_noise = self.nn_model(x_pert, t / timesteps, c=c)
                else:
                    pred_noise = self.nn_model(x_pert, t / timesteps)
                
                # loss is mean squared error between the predicted and true noise
                loss = F.mse_loss(pred_noise, noise)
                loss.backward()
                
                self.optim.step()

            self.gpu_perf.snapshot(f'epoch {ep}')


        # save model periodically
        torch.save(self.nn_model.state_dict(), filename)
        log.info(f"saved model to {filename}")

