import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

class SpriteDataset(Dataset):
    def __init__(self, sfilename, lfilename, transform, target_transform):
        self.sprites = np.load(sfilename)
        self.slabels = np.load(lfilename)
        self.transform = transform
        self.target_transform = target_transform
    
    def __getitem__(self, idx):
        image = self.transform(self.sprites[idx])
        label = self.target_transform(self.slabels[idx])
        return image, label

    def __len__(self):
        return len(self.sprites)

def generate_animation(intermediate_samples, t_steps, fname, n_images_per_row=8):
    intermediate_samples = [make_grid(x, scale_each=True, normalize=True, 
                                      nrow=n_images_per_row).permute(1, 2, 0).numpy() for x in intermediate_samples]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis("off")
    img_plot = ax.imshow(intermediate_samples[0])
    
    def update(frame):
        img_plot.set_array(intermediate_samples[frame])
        ax.set_title(f"T = {t_steps[frame]}")
        fig.tight_layout()
        return img_plot
    
    ani = FuncAnimation(fig, update, frames=len(intermediate_samples), interval=200)
    ani.save(fname)


def get_custom_context(n_samples, n_classes, device):
    """Returns custom context in one-hot encoded form"""
    context = []
    for i in range(n_classes - 1):
        context.extend([i]*(n_samples//n_classes))
    context.extend([n_classes - 1]*(n_samples - len(context)))
    return torch.nn.functional.one_hot(torch.tensor(context), n_classes).float().to(device)
