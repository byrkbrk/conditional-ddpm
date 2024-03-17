from torchvision.utils import save_image
from train_model import TrainModel
from utils import generate_animation, get_custom_context
import os
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Sample images from diffusion model")
    parser.add_argument("--checkpoint-name", type=str, default="checkpoint_0.pth", help="Checkpoint name of diffusion model")
    parser.add_argument("--n-samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--device", type=str, default="cuda", help="GPU device to use")
    parser.add_argument("--n-images-per-row", type=int, default=10, help="Number of images each row contains in the grid")
    args = parser.parse_args()
    
    os.makedirs(os.path.join(os.path.dirname(__file__), "generated-images"), exist_ok=True)
    
    train_model = TrainModel(device=args.device, checkpoint_name=args.checkpoint_name)
    c = get_custom_context(n_samples=args.n_samples, n_classes=train_model.nn_model.n_cfeat, device=train_model.device)
    samples, intermediate_ddpm, t_steps = train_model.sample_ddpm(args.n_samples, c)
    
    save_image(samples, os.path.join(os.path.dirname(__file__), "generated-images", 
                                    f"{train_model.dataset_name}_ddpm_images.jpeg"), 
                                    scale_each=True, normalize=True, nrow=args.n_images_per_row)
    generate_animation(intermediate_ddpm, t_steps, os.path.join(os.path.dirname(__file__), "generated-images",
                                                       f"{train_model.dataset_name}_ani.gif"), 
                                                       n_images_per_row=args.n_images_per_row)
