import torch
import numpy as np
import logging as log
from torch.utils.data import Dataset
import torchvision.transforms as T
from torch import nn

class CustomDataset(Dataset):
    def __init__(self, sfilename, lfilename, null_context=False, augment_horizontal_flip=True):
        self.sprites = np.load(sfilename)
        self.slabels = np.load(lfilename)
        log.info(f"sprite shape: {self.sprites.shape}")
        log.info(f"labels shape: {self.slabels.shape}")

        self.transform = T.Compose([
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.ToTensor(),                # from [0,255] to range [0.0,1.0]
            T.Normalize((0.5,), (0.5,))  # range [-1,1]

        ])

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

