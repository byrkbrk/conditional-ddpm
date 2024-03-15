from sampling_functions import sample_ddpm
from torchvision.utils import save_image
from train_model import TrainModel
from utils import generate_animation
import torch
import os
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Sample images from diffusion model")
    parser.add_argument("--checkpoint-name", type=str, default="checkpoint_0.pth", help="Checkpoint name of diffusion model")
    parser.add_argument("--n-sample", type=int, default=64, help="Number of samples to generate")
    parser.add_argument("--device", type=str, default="cuda", help="GPU device to use")
    args = parser.parse_args()
    
    os.makedirs(os.path.join(os.path.dirname(__file__), "generated-images"), exist_ok=True)
    
    checkpoint = torch.load(os.path.join(os.path.dirname(__file__), "checkpoints", args.checkpoint_name), 
                            map_location=torch.device(args.device))
    nn_model = TrainModel(dataset_name=checkpoint["dataset_name"], 
                          checkpoint_name=args.checkpoint_name, device=args.device).nn_model.eval()
    
    samples, intermediate_ddpm = sample_ddpm(n_sample=args.n_sample, n_channel=nn_model.in_channels,
                                             height=nn_model.height, width=nn_model.width, 
                                             nn_model=nn_model, timesteps=checkpoint["timesteps"],                                              
                                             a_t=checkpoint["a_t"], b_t=checkpoint["b_t"], 
                                             ab_t=checkpoint["ab_t"], device=args.device)
    generate_animation(intermediate_ddpm, os.path.join(os.path.dirname(__file__), "generated-images"))
    save_image(samples, os.path.join(os.path.dirname(__file__), "generated-images", "ddpm_images.jpeg"), 
                scale_each=True, normalize=True)