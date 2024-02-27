import torch
import torch.nn as nn
from torchvision.utils import save_image
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST
from tqdm import tqdm
import os
from models import ContextUnet
from utils import CustomDataset
import numpy as np



class TrainModel(nn.Module):
    def __init__(self,
                 batch_size=100, n_epoch=32, lrate=1e-3,
                 timesteps=500, beta1=1e-4, beta2=0.02, device="cuda", 
                 dataset_name=None, checkpoint_name=None):
        super(TrainModel, self).__init__()
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.lrate = lrate
        self.timesteps = timesteps
        self.beta1 = beta1
        self.beta2 = beta2
        self.device = torch.device(device)
        self.dataset_name = dataset_name
        self.file_dir = os.path.dirname(__file__)
        self.checkpoint_name = checkpoint_name
        self.nn_model = self.initialize_nn_model(dataset_name, checkpoint_name, self.file_dir)
        self.create_dirs(self.file_dir)

    def train(self):
        # ddpm noise schedule
        b_t = (self.beta2 - self.beta1)*torch.linspace(
            0, 1, self.timesteps+1, device=self.device) + self.beta1
        a_t = 1 - b_t
        ab_t = torch.cumprod(a_t, dim=0)
        
        dataset = self.get_dataset(self.dataset_name, 
                            self.get_transforms(self.dataset_name))
        dataloader = DataLoader(dataset, self.batch_size, True)
        optim = torch.optim.Adam(self.nn_model.parameters(), lr=self.lrate)
        self.nn_model.to(self.device)

        for epoch in range(self.n_epoch):
            ave_loss = 0

            for x, c in tqdm(dataloader, mininterval=2, desc=f"Epoch {epoch}"):
                x = x.to(self.device)
                c = c.to(self.device).squeeze()
                
                # randomly mask out c
                context_mask = torch.bernoulli(torch.zeros(c.shape[0]) + 0.9).to(self.device)
                c = c * context_mask.unsqueeze(-1)
                
                # perturb data
                noise = torch.randn_like(x)
                t = torch.randint(1, self.timesteps + 1, (x.shape[0], )).to(self.device)
                x_pert = self.perturb_input(x, t, noise, ab_t)

                # predict noise
                pred_noise = self.nn_model(x_pert, t / self.timesteps, c=c)

                # obtain loss
                loss = torch.nn.functional.mse_loss(pred_noise, noise)
                
                # update params
                optim.zero_grad()
                loss.backward()
                optim.step()

                ave_loss += loss.item()/len(dataloader)
            print(f"Epoch: {epoch}, loss: {ave_loss}")
            save_image(x, os.path.join(self.file_dir, "saved-images", f"x_orig_{epoch}.jpeg"))
            save_image(self.get_x_unpert(x_pert, t, pred_noise, ab_t), 
                       os.path.join(self.file_dir, "saved-images", f"x_denoised_{epoch}.jpeg"))
            self.save_checkpoint(self.nn_model, optim, epoch, ave_loss, 
                                 self.timesteps, a_t, b_t, ab_t,
                                 self.device, self.dataset_name, self.file_dir)


    def perturb_input(self, x, t, noise, ab_t):
        return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]) * noise
    
    def get_dataset(self, dataset_name, transforms):
        assert dataset_name in {"mnist", "fashion_mnist", "sprite"}, "Unknown dataset"
        
        transform, target_transform = transforms
        if dataset_name=="mnist":
            return MNIST(".", True, transform, target_transform, True)
        if dataset_name=="fashion_mnist":
            return FashionMNIST(".", True, transform, target_transform, True)
        if dataset_name=="sprite":
            return CustomDataset("sprites_1788_16x16.npy", "sprite_labels_nc_1788_16x16.npy", transform)

    def get_transforms(self, dataset_name):
        assert dataset_name in {"mnist", "fashion_mnist", "sprite"}, "Unknown dataset"

        if dataset_name in {"mnist", "fashion_mnist"}:
            transform = transforms.Compose([
                transforms.ToTensor(),
                lambda x: 2*(x - 0.5)
            ])
            target_transform = transforms.Compose([
                lambda x: torch.tensor([x]),
                lambda class_labels, n_classes=10: nn.functional.one_hot(
                class_labels, n_classes)
            ])

        if dataset_name=="sprite":
            transform = transforms.Compose([
                transforms.ToTensor(),                # from [0,255] to range [0.0,1.0]
                transforms.Normalize((0.5,), (0.5,))  # range [-1,1]
            ])
            target_transform = None    
        return transform, target_transform
    
    def get_x_unpert(self, x_pert, t, pred_noise, ab_t):
        return (x_pert - (1 - ab_t[t, None, None, None]) * pred_noise) / ab_t.sqrt()[t, None, None, None]
    
    def initialize_nn_model(self, dataset_name, checkpoint_name, file_dir):
        """Returns the instantiated model based on dataset name"""
        assert dataset_name in {"mnist", "fashion_mnist", "sprite"}, "Unknown dataset name"

        if dataset_name in {"mnist", "fashion_mnist"}:
            nn_model = ContextUnet(in_channels=1, n_feat=64, n_cfeat=10, height=28)
        
        if dataset_name=="sprite":
            nn_model = ContextUnet(in_channels=3, n_feat=64, n_cfeat=5, height=16)
        
        if checkpoint_name:
            checkpoint = torch.load(os.path.join(file_dir, "checkpoints", checkpoint_name))
            nn_model.to(checkpoint["device"])
            nn_model.load_state_dict(checkpoint["model_state_dict"])
        return nn_model

    def save_checkpoint(self, model, optimizer, epoch, loss, 
                        timesteps, a_t, b_t, ab_t, device,
                        dataset_name, file_dir):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": loss,
            "timesteps": timesteps, 
            "a_t": a_t, 
            "b_t": b_t, 
            "ab_t": ab_t,
            "device": device,
            "dataset_name": dataset_name
        }
        torch.save(checkpoint, os.path.join(
            file_dir, "checkpoints", f"checkpoint_{epoch}.pth"))

    def create_dirs(self, file_dir):
        dir_names = ["checkpoints", "saved-images"]
        for dir_name in dir_names:
            os.makedirs(os.path.join(file_dir, dir_name), exist_ok=True)

    @torch.no_grad()
    def sample_ddpm(self, n_sample=64, dataset_name="mnist", context=None, save_rate=20):
        checkpoint = ""
        timesteps, a_t, b_t, ab_t, device = list(checkpoint.values)[-5:]
        nn_model = self.nn_model
        
        assert dataset_name in {"mnist", "fashion_mnist", "sprite"}, "Unknown dataset name"

        if dataset_name in {"mnist", "fashion_mnist"}:
            n_channels, height, width = 3, 28, 28
        if dataset_name in {"sprite"}:
            n_channels, height, width = 1, 16, 16
        
        # x_T ~ N(0, 1), sample initial noise
        samples = torch.randn(n_sample, n_channels, height, width).to(self.device)  

        # array to keep track of generated steps for plotting
        intermediate = [] 
        for i in range(timesteps, 0, -1):
            print(f'sampling timestep {i:3d}', end='\r')

            # reshape time tensor
            t = torch.tensor([i / timesteps])[:, None, None, None].to(device)

            # sample some random noise to inject back in. For i = 1, don't add back in noise
            z = torch.randn_like(samples) if i > 1 else 0

            eps = nn_model(samples, t, c=context)    # predict noise e_(x_t,t)
            samples = self.denoise_add_noise(samples, i, eps, a_t, b_t, ab_t, z)
            if i % save_rate ==0 or i==timesteps or i<8:
                intermediate.append(samples.detach().cpu().numpy())

        intermediate = np.stack(intermediate)
        return samples, intermediate
    
    def denoise_add_noise(self, x, t, pred_noise, a_t, b_t, ab_t, z=None):
        if z is None:
            z = torch.randn_like(x)
        noise = b_t.sqrt()[t] * z
        mean = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
        return mean + noise
