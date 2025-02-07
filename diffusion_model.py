import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from tqdm import tqdm
import os
from models import ContextUnet
from utils import SpriteDataset, generate_animation
import matplotlib.pyplot as plt



class DiffusionModel(nn.Module):
    def __init__(self, device=None, dataset_name=None, checkpoint_name=None):
        super(DiffusionModel, self).__init__()
        self.device = self.initialize_device(device)
        self.file_dir = os.path.dirname(__file__)
        self.dataset_name = self.initialize_dataset_name(self.file_dir, checkpoint_name, dataset_name)
        self.checkpoint_name = checkpoint_name
        self.nn_model = self.initialize_nn_model(self.dataset_name, checkpoint_name, self.file_dir, self.device)
        self.create_dirs(self.file_dir)

    def train(self, batch_size=64, n_epoch=32, lr=1e-3, timesteps=500, beta1=1e-4, beta2=0.02,
          checkpoint_save_dir=None, image_save_dir=None):
        """Trains model for given inputs"""
        self.nn_model.train()
        _, _, ab_t = self.get_ddpm_noise_schedule(timesteps, beta1, beta2, self.device)

        start_epoch = self.get_start_epoch(self.checkpoint_name, self.file_dir)

        if start_epoch >= 50:
            self.dataset_name = "custom" 

        dataset = self.instantiate_dataset(self.dataset_name, 
                            self.get_transforms(self.dataset_name), self.file_dir)
        dataloader = self.initialize_dataloader(dataset, batch_size, self.checkpoint_name, self.file_dir)

        optim = self.initialize_optimizer(self.nn_model, lr, self.checkpoint_name, self.file_dir, self.device)
        scheduler = self.initialize_scheduler(optim, self.checkpoint_name, self.file_dir, self.device)

        for epoch in range(start_epoch, start_epoch + n_epoch):
            ave_loss = 0

            for x, c in tqdm(dataloader, mininterval=2, desc=f"Epoch {epoch}"):
                x = x.to(self.device)
                c = self.get_masked_context(c).to(self.device)
                
                noise = torch.randn_like(x)
                t = torch.randint(1, timesteps + 1, (x.shape[0], )).to(self.device)
                x_pert = self.perturb_input(x, t, noise, ab_t)

                pred_noise = self.nn_model(x_pert, t / timesteps, c=c)

                loss = torch.nn.functional.mse_loss(pred_noise, noise)
                
                optim.zero_grad()
                loss.backward()
                optim.step()

                ave_loss += loss.item() / len(dataloader)

            scheduler.step()
            print(f"Epoch: {epoch}, loss: {ave_loss}")

            sample_size = min(8, x.shape[0]) 
            self.save_tensor_images(
                x[:sample_size],  
                x_pert[:sample_size],  
                self.get_x_unpert(x_pert[:sample_size], t[:sample_size], pred_noise[:sample_size], ab_t), 
                c[:sample_size],  
                epoch, 
                self.file_dir,
                image_save_dir
            )

            self.save_checkpoint(
                self.nn_model, optim, scheduler, epoch, ave_loss, 
                timesteps, beta1, beta2, self.device, self.dataset_name,
                dataloader.batch_size, self.file_dir, checkpoint_save_dir
            )


    @torch.no_grad()
    def sample_ddpm(self, n_samples, context=None, timesteps=None, 
                    beta1=None, beta2=None, save_rate=20, inference_transform=lambda x: (x+1)/2):
        """Returns the final denoised sample x0,
        intermediate samples xT, xT-1, ..., x1, and
        times tT, tT-1, ..., t1
        """
        if all([timesteps, beta1, beta2]):
            a_t, b_t, ab_t = self.get_ddpm_noise_schedule(timesteps, beta1, beta2, self.device)
        else:
            timesteps, a_t, b_t, ab_t = self.get_ddpm_params_from_checkpoint(self.file_dir,
                                                                             self.checkpoint_name, 
                                                                             self.device)
        
        self.nn_model.eval()
        samples = torch.randn(n_samples, self.nn_model.in_channels, 
                              self.nn_model.height, self.nn_model.width, 
                              device=self.device)
        intermediate_samples = [samples.detach().cpu()] # samples at T = timesteps
        t_steps = [timesteps] # keep record of time to use in animation generation
        for t in range(timesteps, 0, -1):
            print(f"Sampling timestep {t}", end="\r")
            if t % 50 == 0: print(f"Sampling timestep {t}")

            z = torch.randn_like(samples) if t > 1 else 0
            pred_noise = self.nn_model(samples, 
                                       torch.tensor([t/timesteps], device=self.device)[:, None, None, None], 
                                       context)
            samples = self.denoise_add_noise(samples, t, pred_noise, a_t, b_t, ab_t, z)
            
            if t % save_rate == 1 or t < 8:
                intermediate_samples.append(inference_transform(samples.detach().cpu()))
                t_steps.append(t-1)
        return intermediate_samples[-1], intermediate_samples, t_steps

    def perturb_input(self, x, t, noise, ab_t):
        """Perturbs given input
        i.e., Algorithm 1, step 5, argument of epsilon_theta in the article
        """
        return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]).sqrt() * noise
    
    def instantiate_dataset(self, dataset_name, transforms, file_dir, train=True):
        """Returns instantiated dataset for given dataset name"""
        assert dataset_name in {"mnist", "fashion_mnist", "sprite", "cifar10", "custom"}, "Unknown dataset"

        transform, target_transform = transforms
        if dataset_name == "mnist":
            return MNIST(os.path.join(file_dir, "datasets"), train, transform, target_transform, True)
        if dataset_name == "fashion_mnist":
            return FashionMNIST(os.path.join(file_dir, "datasets"), train, transform, target_transform, True)
        if dataset_name == "sprite":
            return SpriteDataset(os.path.join(file_dir, "datasets"), transform, target_transform)
        if dataset_name == "cifar10":
            return CIFAR10(os.path.join(file_dir, "datasets"), train, transform, target_transform, True)
        if dataset_name == "custom":
            from torch.utils.data import Dataset

            class CustomDataset(Dataset):
                def __init__(self, data_dir, transform=None, target_transform=None):
                    self.data_dir = data_dir
                    self.transform = transform
                    self.target_transform = target_transform
                    self.images = []
                    self.labels = []

                    for label in range(10):  # Assuming 10 classes (0-9)
                        class_dir = os.path.join(data_dir, str(label))
                        for file_name in os.listdir(class_dir):
                            if file_name.endswith(".png") or file_name.endswith(".jpg"):
                                self.images.append(os.path.join(class_dir, file_name))
                                self.labels.append(label)

                def __len__(self):
                    return len(self.images)

                def __getitem__(self, idx):
                    from PIL import Image

                    img_path = self.images[idx]
                    label = self.labels[idx]

                    image = Image.open(img_path).convert("RGB")
                    if self.transform:
                        image = self.transform(image)
                    if self.target_transform:
                        label = self.target_transform(label)

                    return image, label

            file_dir = os.path.abspath(os.path.join(file_dir, "..", "..", "pipelines"))
            return CustomDataset(os.path.join(file_dir, "preprocessed_images", "best_combination"), transform, target_transform)

    def get_transforms(self, dataset_name):
        """Returns transform and target-transform for given dataset name"""
        assert dataset_name in {"mnist", "fashion_mnist", "sprite", "cifar10", "custom"}, "Unknown dataset"

        if dataset_name in {"mnist", "fashion_mnist", "cifar10", "custom"}:
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((28, 28)),                
                transforms.ToTensor()
            ])
            target_transform = transforms.Compose([
                lambda x: torch.tensor([x]),
                lambda class_labels, n_classes=10: nn.functional.one_hot(class_labels, n_classes).squeeze()
            ])

        if dataset_name == "sprite":
            transform = transforms.Compose([
                transforms.ToTensor(),  # from [0,255] to range [0.0,1.0]
                lambda x: 2 * x - 1       # range [-1,1]
            ])
            target_transform = lambda x: torch.from_numpy(x).to(torch.float32)

        return transform, target_transform
    
    def get_x_unpert(self, x_pert, t, pred_noise, ab_t):
        """Removes predicted noise pred_noise from perturbed image x_pert"""
        return (x_pert - (1 - ab_t[t, None, None, None]).sqrt() * pred_noise) / ab_t.sqrt()[t, None, None, None]
    
    def initialize_nn_model(self, dataset_name, checkpoint_name, file_dir, device):
        """Returns the instantiated model based on dataset name"""
        assert dataset_name in {"mnist", "fashion_mnist", "sprite", "cifar10", "custom"}, "Unknown dataset name"

        if dataset_name in {"mnist", "fashion_mnist"}:
            nn_model = ContextUnet(in_channels=1, height=28, width=28, n_feat=64, n_cfeat=10, n_downs=2)
        elif dataset_name == "sprite":
            nn_model = ContextUnet(in_channels=3, height=16, width=16, n_feat=64, n_cfeat=5, n_downs=2)
        elif dataset_name == "cifar10":
            nn_model = ContextUnet(in_channels=3, height=32, width=32, n_feat=64, n_cfeat=10, n_downs=4)
        elif dataset_name == "custom":
            nn_model = ContextUnet(
            in_channels=1,  
            height=28,     
            width=28,      
            n_feat=64, 
            n_cfeat=10, 
            n_downs=2      
        )

        if checkpoint_name:
            checkpoint = torch.load(os.path.join(file_dir, "checkpoints", checkpoint_name), map_location=device)
            nn_model.to(device)
            nn_model.load_state_dict(checkpoint["model_state_dict"])
            return nn_model

        return nn_model.to(device)

    def save_checkpoint(self, model, optimizer, scheduler, epoch, loss, 
                        timesteps, beta1, beta2, device, dataset_name, batch_size, 
                        file_dir, save_dir):
        """Saves checkpoint for given variables"""
        if save_dir is None:
            fpath = os.path.join(file_dir, "checkpoints", f"{dataset_name}_checkpoint_{epoch}.pth")
        else:
            fpath = os.path.join(save_dir, f"{dataset_name}_checkpoint_{epoch}.pth")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": loss,
            "timesteps": timesteps, 
            "beta1": beta1, 
            "beta2": beta2,
            "device": device,
            "dataset_name": dataset_name,
            "batch_size": batch_size
        }
        torch.save(checkpoint, fpath)

    def create_dirs(self, file_dir):
        """Creates directories required for training"""
        dir_names = ["checkpoints", "saved-images"]
        for dir_name in dir_names:
            os.makedirs(os.path.join(file_dir, dir_name), exist_ok=True)

    def initialize_optimizer(self, nn_model, lr, checkpoint_name, file_dir, device):
        """Instantiates and initializes the optimizer based on checkpoint availability"""
        optim = torch.optim.Adam(nn_model.parameters(), lr=lr)
        if checkpoint_name:
            checkpoint = torch.load(os.path.join(file_dir, "checkpoints", checkpoint_name), map_location=device)
            optim.load_state_dict(checkpoint["optimizer_state_dict"])
        return optim

    def initialize_scheduler(self, optimizer, checkpoint_name, file_dir, device):
        """Instantiates and initializes scheduler based on checkpoint availability"""
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, 
                                                      end_factor=0.01, total_iters=50)
        if checkpoint_name:
            checkpoint = torch.load(os.path.join(file_dir, "checkpoints", checkpoint_name), map_location=device)
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return scheduler
    
    def get_start_epoch(self, checkpoint_name, file_dir):
        """Returns starting epoch for training"""
        if checkpoint_name:
            start_epoch = torch.load(os.path.join(file_dir, "checkpoints", checkpoint_name), 
                                    map_location=torch.device("cpu"))["epoch"] + 1
        else:
            start_epoch = 0
        return start_epoch

    def save_tensor_images(self, x_orig, x_noised, x_denoised, labels, cur_epoch, file_dir, save_dir=None):
        """
        Saves all digits (0-9) in one combined image with titles indicating their labels.
        """
        if save_dir is None:
            save_dir = os.path.join("./saved-images", f"epoch_{cur_epoch}")
        os.makedirs(save_dir, exist_ok=True)
        
        inference_transform = lambda x: (x + 1) / 2

        if isinstance(labels, torch.Tensor) and labels.dim() > 1:
            labels = labels.argmax(dim=1)  
        elif isinstance(labels, torch.Tensor) and labels.dim() == 0:
            labels = labels.unsqueeze(0)  

        fig, axes = plt.subplots(nrows=3, ncols=10, figsize=(15, 5)) 
        fig.suptitle(f"Epoch {cur_epoch}: MNIST Digits", fontsize=16)

        for i in range(10):
            indices = (labels == i).nonzero(as_tuple=True)[0]
            if len(indices) > 0: 
                idx = indices[0]  
                orig = inference_transform(x_orig[idx].detach().cpu())
                noised = inference_transform(x_noised[idx].detach().cpu())
                denoised = inference_transform(x_denoised[idx].detach().cpu())

                axes[0, i].imshow(orig.permute(1, 2, 0), cmap="gray")
                axes[0, i].set_title(f"Label {i}")
                axes[0, i].axis("off")

                axes[1, i].imshow(noised.permute(1, 2, 0), cmap="gray")
                axes[1, i].axis("off")

                axes[2, i].imshow(denoised.permute(1, 2, 0), cmap="gray")
                axes[2, i].axis("off")
            else:
                for j in range(3):
                    axes[j, i].axis("off")

        save_path = os.path.join(save_dir, f"mnist_digits_epoch_{cur_epoch}.png")
        plt.tight_layout()
        plt.subplots_adjust(top=0.85) 
        plt.savefig(save_path)
        plt.close(fig)



    def get_ddpm_noise_schedule(self, timesteps, beta1, beta2, device):
        """Returns ddpm noise schedule variables, a_t, b_t, ab_t
        b_t: \beta_t
        a_t: \alpha_t
        ab_t \bar{\alpha}_t
        """
        b_t = torch.linspace(beta1, beta2, timesteps+1, device=device)
        a_t = 1 - b_t
        ab_t = torch.cumprod(a_t, dim=0)
        return a_t, b_t, ab_t
    
    def get_ddpm_params_from_checkpoint(self, file_dir, checkpoint_name, device):
        """Returns scheduler variables T, a_t, ab_t, and b_t from checkpoint"""
        checkpoint = torch.load(os.path.join(file_dir, "checkpoints", checkpoint_name), torch.device("cpu"))
        T = checkpoint["timesteps"]
        a_t, b_t, ab_t = self.get_ddpm_noise_schedule(T, checkpoint["beta1"], checkpoint["beta2"], device)
        return T, a_t, b_t, ab_t
    
    def denoise_add_noise(self, x, t, pred_noise, a_t, b_t, ab_t, z):
        """Removes predicted noise from x and adds gaussian noise z
        i.e., Algorithm 2, step 4 at the ddpm article
        """
        noise = b_t.sqrt()[t]*z
        denoised_x = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
        return denoised_x + noise
    
    def initialize_dataset_name(self, file_dir, checkpoint_name, dataset_name):
        """Initializes dataset name based on checkpoint availability"""
        if checkpoint_name:
            return torch.load(os.path.join(file_dir, "checkpoints", checkpoint_name), 
                                    map_location=torch.device("cpu"))["dataset_name"]
        return dataset_name
    
    def initialize_dataloader(self, dataset, batch_size, checkpoint_name, file_dir):
        """Returns dataloader based on batch-size of checkpoint if present"""
        if checkpoint_name:
            batch_size = torch.load(os.path.join(file_dir, "checkpoints", checkpoint_name), 
                                    map_location=torch.device("cpu"))["batch_size"]
        return DataLoader(dataset, batch_size, True)
    
    def get_masked_context(self, context, p=0.9):
        "Randomly mask out context"
        return context*torch.bernoulli(torch.ones((context.shape[0], 1))*p)
    
    def save_generated_samples_into_folder(self, n_samples, context, folder_path, **kwargs):
        """Save DDPM generated inputs into a specified directory"""
        samples, _, _ = self.sample_ddpm(n_samples, context, **kwargs)
        for i, sample in enumerate(samples):
            save_image(sample, os.path.join(folder_path, f"image_{i}.jpeg"))
    
    def save_dataset_test_images(self, n_samples):
        """Save dataset test images with specified number"""
        folder_path = os.path.join(self.file_dir, f"{self.dataset_name}-test-images")
        os.makedirs(folder_path, exist_ok=True)

        dataset = self.instantiate_dataset(self.dataset_name, 
                            (transforms.ToTensor(), None), self.file_dir, train=False)
        dataloader = DataLoader(dataset, 1, True)
        for i, (image, _) in enumerate(dataloader):
            if i == n_samples: break
            save_image(image, os.path.join(folder_path, f"image_{i}.jpeg"))

    def initialize_device(self, device):
        """Initializes device based on availability"""
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        return torch.device(device)
    
    def get_custom_context(self, n_samples, n_classes, device):
        """Returns custom context in one-hot encoded form"""
        context = []
        for i in range(n_classes - 1):
            context.extend([i]*(n_samples//n_classes))
        context.extend([n_classes - 1]*(n_samples - len(context)))
        return torch.nn.functional.one_hot(torch.tensor(context), n_classes).float().to(device)
    
    def generate(self, n_samples, n_images_per_row, timesteps, beta1, beta2):
        """Generates x0 and intermediate samples xi via DDPM, 
        and saves as jpeg and gif files for given inputs
        """
        root = os.path.join(self.file_dir, "generated-images")
        os.makedirs(root, exist_ok=True)
        x0, intermediate_samples, t_steps = self.sample_ddpm(n_samples,
                                                             self.get_custom_context(
                                                                 n_samples, self.nn_model.n_cfeat, 
                                                                 self.device),
                                                             timesteps,
                                                             beta1,
                                                             beta2,)
        save_image(x0, os.path.join(root, f"{self.dataset_name}_ddpm_images.jpeg"), nrow=n_images_per_row)
        generate_animation(intermediate_samples,
                           t_steps, 
                           os.path.join(root, f"{self.dataset_name}_ani.gif"),
                           n_images_per_row)
