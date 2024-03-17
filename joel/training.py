import torch
import logging as log
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from gpuperf import *
from tqdm import tqdm

class CustomDataset(Dataset):
    def __init__(self, sfilename, lfilename, transform, null_context=False):
        self.sprites = np.load(sfilename)
        self.slabels = np.load(lfilename)
        print(f"sprite shape: {self.sprites.shape}")
        print(f"labels shape: {self.slabels.shape}")
        self.transform = transform
        self.null_context = null_context
        self.sprites_shape = self.sprites.shape
        self.slabel_shape = self.slabels.shape
                
    # Return the number of images in the dataset
    def __len__(self):
        return len(self.sprites)
    
    # Get the image and label at a given index
    def __getitem__(self, idx):
        # Return the image and label as a tuple
        if self.transform:
            image = self.transform(self.sprites[idx])
            if self.null_context:
                label = torch.tensor(0).to(torch.int64)
            else:
                label = torch.tensor(self.slabels[idx]).to(torch.int64)
        return (image, label)

    def getshapes(self):
        # return shapes of data and labels
        return self.sprites_shape, self.slabel_shape
    
class Training():
    def __init__(self, noise_sampler, nn_model, lr, batch_size, device, gpu_perf):
        self.noise_sampler = noise_sampler
        self.nn_model = nn_model
        self.lr = lr
        self.device = device
        self.gpu_perf = gpu_perf
        
        transform = transforms.Compose([
            transforms.ToTensor(),                # from [0,255] to range [0.0,1.0]
            transforms.Normalize((0.5,), (0.5,))  # range [-1,1]

        ])

        # load dataset and construct optimizer
        dataset = CustomDataset("./sprites_1788_16x16.npy", "./sprite_labels_nc_1788_16x16.npy", transform, null_context=False)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
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

