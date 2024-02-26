from sampling_functions import sample_ddpm
from diffusion_utilities import plot_sample
from torchvision.utils import save_image
from train_model import TrainModel
import torch
import os
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Sample images from diffusion model")
    parser.add_argument("--checkpoint-name", type=str, default="checkpoint_2.pth", help="Checkpoint name of diffusion model")
    parser.add_argument("--n-sample", type=int, default=64, help="Number of samples to generate")
    args = parser.parse_args()
    
    checkpoint_name = args.checkpoint_name
    n_sample = args.n_sample
    os.makedirs(os.path.join(os.path.dirname(__file__), "generated-images"),
                 exist_ok=True)
    
    checkpoint = torch.load(os.path.join("checkpoints", checkpoint_name))
    nn_model = TrainModel(dataset_name=checkpoint["dataset_name"], checkpoint_name=checkpoint_name).nn_model
    samples, intermediate_ddpm = sample_ddpm(n_sample=n_sample, n_channel=nn_model.in_channels,
                                             height=nn_model.h, width=nn_model.h, 
                                             nn_model=nn_model, timesteps=checkpoint["timesteps"],                                              
                                             a_t=checkpoint["a_t"], b_t=checkpoint["b_t"], 
                                             ab_t=checkpoint["ab_t"], device=checkpoint["device"],)
    plot_sample(intermediate_ddpm, n_sample, 8, "generated-images/", 
                "ani_run", None, True)
    save_image(samples, os.path.join("generated-images", "ddpm_images.jpeg"))