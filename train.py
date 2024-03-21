import argparse
from diffusion_model import DiffusionModel

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Train diffusion model")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training")
    parser.add_argument("--dataset-name", type=str, default="mnist", help="Dataset name for training")
    parser.add_argument("--device", type=str, default="cuda", help="Device for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for training")
    parser.add_argument("--checkpoint-name", type=str, default=None, help="Checkpoint name for pre-training")
    args = parser.parse_args()

    diffusion_model = DiffusionModel(device=args.device, dataset_name=args.dataset_name, 
                             checkpoint_name=args.checkpoint_name)
    diffusion_model.train(batch_size=args.batch_size, n_epoch=args.epochs, lr=args.lr)


