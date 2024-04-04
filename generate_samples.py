from diffusion_model import DiffusionModel
from utils import get_custom_context
import os
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and save indiviaul images")
    parser.add_argument("checkpoint_name", type=str, default=None, help="Checkpoint name of diffusion model")
    parser.add_argument("--n-samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--device", type=str, default="cuda", help="GPU device to use")
    parser.add_argument("--save-test-dataset", type=bool, default=False, help="Save test dataset into folder")
    args = parser.parse_args()
    
    # define folder path & create into which generated images be saved
    folder_path = os.path.join(os.path.dirname(__file__), os.path.splitext(args.checkpoint_name)[0])
    os.makedirs(folder_path)
    
    # save images
    diffusion_model = DiffusionModel(args.device, None, args.checkpoint_name)
    diffusion_model.save_generated_samples_into_folder(args.n_samples,
                                                       get_custom_context(args.n_samples, 
                                                                          diffusion_model.nn_model.n_cfeat, 
                                                                          diffusion_model.device),
                                                       folder_path)
    if args.save_test_dataset:
        diffusion_model.save_dataset_test_images(args.n_samples)

