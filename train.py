import argparse
from train_model import TrainModel

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Train diffusion model")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=32, help="Number of epochs for training")
    parser.add_argument("--dataset-name", type=str, default="mnist", help="Dataset name for training")
    parser.add_argument("--device", type=str, default="cuda", help="Device for training")
    parser.add_argument("--lrate", type=float, default=1e-3, help="Learning rate for training")
    args = parser.parse_args()

    train_model = TrainModel(batch_size=args.batch_size, n_epoch=args.epochs, 
                             device=args.device, dataset_name=args.dataset_name,
                             lrate=args.lrate)
    train_model.train()


