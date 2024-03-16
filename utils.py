import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

class CustomDataset(Dataset):
    def __init__(self, sfilename, lfilename, transform, null_context=False):
        self.sprites = np.load(sfilename)
        self.slabels = np.load(lfilename)
        #print(f"sprite shape: {self.sprites.shape}")
        #print(f"labels shape: {self.slabels.shape}")
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
    

def generate_animation(intermediate_samples, fname, n_images_per_row=8):
    intermediate_samples = [make_grid(x, scale_each=True, normalize=True, 
                                      nrow=n_images_per_row).permute(1, 2, 0).numpy() for x in intermediate_samples]
    fig, ax = plt.subplots()
    img_plot = ax.imshow(intermediate_samples[0])
    
    def update(frame):
        img_plot.set_array(intermediate_samples[frame])
        return img_plot
    
    ani = FuncAnimation(fig, update, frames=len(intermediate_samples), interval=200)
    ani.save(fname)


def get_custom_context(n_samples, n_classes, device):
    context = []
    for i in range(n_classes - 1):
        context.extend([i]*(n_samples//n_classes))
    context.extend([n_classes - 1]*(n_samples - len(context)))
    return torch.nn.functional.one_hot(torch.tensor(context), n_classes).float().to(device)
    

if __name__ == "__main__":
    n_samples = 30
    n_classes = 10
    device = torch.device("mps")
    c = get_custom_context(n_samples, n_classes, device)
    print(c.shape)
    print(c)