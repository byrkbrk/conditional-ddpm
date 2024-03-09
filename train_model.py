import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST
from tqdm import tqdm
import os
from models import ContextUnet
from utils import CustomDataset



class TrainModel(nn.Module):
    def __init__(self, device="cuda", dataset_name=None, checkpoint_name=None):
        super(TrainModel, self).__init__()
        self.device = torch.device(device)
        self.dataset_name = dataset_name
        self.file_dir = os.path.dirname(__file__)
        self.checkpoint_name = checkpoint_name
        self.nn_model = self.initialize_nn_model(dataset_name, checkpoint_name, self.file_dir, self.device)
        self.create_dirs(self.file_dir)

    def train(self, batch_size=64, n_epoch=32, lr=1e-3, timesteps=500, beta1=1e-4, beta2=0.02):
        # noise schedule
        a_t ,b_t, ab_t = self.get_ddpm_noise_schedule(timesteps, beta1, beta2, self.device)
        
        dataset = self.get_dataset(self.dataset_name, 
                            self.get_transforms(self.dataset_name), self.file_dir)
        dataloader = DataLoader(dataset, batch_size, True)
        optim = self.initialize_optimizer(self.nn_model, lr, self.checkpoint_name, self.file_dir, self.device)
        scheduler = self.initialize_scheduler(optim, self.checkpoint_name, self.file_dir, self.device)

        for epoch in range(self.get_start_epoch(self.checkpoint_name, self.file_dir), 
                           self.get_start_epoch(self.checkpoint_name, self.file_dir) + n_epoch):
            ave_loss = 0

            for x, c in tqdm(dataloader, mininterval=2, desc=f"Epoch {epoch}"):
                x = x.to(self.device)
                c = c.to(self.device).squeeze()
                
                # randomly mask out c
                context_mask = torch.bernoulli(torch.zeros(c.shape[0]) + 0.9).to(self.device)
                c = c * context_mask.unsqueeze(-1)
                
                # perturb data
                noise = torch.randn_like(x)
                t = torch.randint(1, timesteps + 1, (x.shape[0], )).to(self.device)
                x_pert = self.perturb_input(x, t, noise, ab_t)

                # predict noise
                pred_noise = self.nn_model(x_pert, t / timesteps, c=c)

                # obtain loss
                loss = torch.nn.functional.mse_loss(pred_noise, noise)
                
                # update params
                optim.zero_grad()
                loss.backward()
                optim.step()

                ave_loss += loss.item()/len(dataloader)
            scheduler.step()
            print(f"Epoch: {epoch}, loss: {ave_loss}")
            self.save_tensor_images(x, x_pert, self.get_x_unpert(x_pert, t, pred_noise, ab_t), epoch, self.file_dir)
            self.save_checkpoint(self.nn_model, optim, scheduler, epoch, ave_loss, 
                                 timesteps, a_t, b_t, ab_t, self.device,
                                 self.dataset_name, self.file_dir)


    def perturb_input(self, x, t, noise, ab_t):
        return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]) * noise
    
    def get_dataset(self, dataset_name, transforms, file_dir):
        assert dataset_name in {"mnist", "fashion_mnist", "sprite"}, "Unknown dataset"
        
        transform, target_transform = transforms
        if dataset_name=="mnist":
            return MNIST(file_dir, True, transform, target_transform, True)
        if dataset_name=="fashion_mnist":
            return FashionMNIST(file_dir, True, transform, target_transform, True)
        if dataset_name=="sprite":
            return CustomDataset(os.path.join(file_dir, "sprites_1788_16x16.npy"), 
                                 os.path.join(file_dir, "sprite_labels_nc_1788_16x16.npy"), 
                                 transform)

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
    
    def initialize_nn_model(self, dataset_name, checkpoint_name, file_dir, device):
        """Returns the instantiated model based on dataset name"""
        assert dataset_name in {"mnist", "fashion_mnist", "sprite"}, "Unknown dataset name"

        if dataset_name in {"mnist", "fashion_mnist"}:
            nn_model = ContextUnet(in_channels=1, n_feat=64, n_cfeat=10, height=28)
        
        if dataset_name=="sprite":
            nn_model = ContextUnet(in_channels=3, n_feat=64, n_cfeat=5, height=16)
        
        if checkpoint_name:
            checkpoint = torch.load(os.path.join(file_dir, "checkpoints", checkpoint_name), map_location=device)
            nn_model.to(device)
            nn_model.load_state_dict(checkpoint["model_state_dict"])
            return nn_model
        return nn_model.to(device)

    def save_checkpoint(self, model, optimizer, scheduler, epoch, loss, 
                        timesteps, a_t, b_t, ab_t, device, dataset_name,
                        file_dir):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
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

    def initialize_optimizer(self, nn_model, lr, checkpoint_name, file_dir, device):
        optim = torch.optim.Adam(nn_model.parameters(), lr=lr)
        if checkpoint_name:
            checkpoint = torch.load(os.path.join(file_dir, "checkpoints", checkpoint_name), map_location=device)
            optim.load_state_dict(checkpoint["optimizer_state_dict"])
        return optim

    def initialize_scheduler(self, optimizer, checkpoint_name, file_dir, device):
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.01, 
                                                    total_iters=32)
        if checkpoint_name:
            checkpoint = torch.load(os.path.join(file_dir, "checkpoints", checkpoint_name), map_location=device)
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return scheduler
    
    def get_start_epoch(self, checkpoint_name, file_dir):
        if checkpoint_name:
            start_epoch = torch.load(os.path.join(file_dir, "checkpoints", checkpoint_name), 
                                    map_location=torch.device("cpu"))["epoch"] + 1
        else:
            start_epoch = 0
        return start_epoch
    
    def save_tensor_images(self, x_orig, x_noised, x_denoised, cur_epoch, file_dir):
        save_image([make_grid(x_orig), make_grid(x_noised), make_grid(x_denoised)],
                   os.path.join(file_dir, "saved-images", f"x_orig_noised_denoised_{cur_epoch}.jpeg"))

    def get_ddpm_noise_schedule(self, timesteps, beta1, beta2, device):
        # ddpm noise schedule
        b_t = (beta2 - beta1)*torch.linspace(0, 1, timesteps+1, device=device) + beta1
        a_t = 1 - b_t
        ab_t = torch.cumprod(a_t, dim=0)
        return a_t, b_t, ab_t