""" Main script for training a video classification model on HMDB51 dataset. """

import argparse
from typing import Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from datasets.TSNDataset import TSNDataset
from models import model_creator
from train import create_dataloaders, create_optimizer, evaluate
from utils import statistics
from utils.early_stopping import EarlyStopping


def aggregate_scores(scores, segments, method='mean'):
    total, num_classes = scores.shape
    num_batches = total // segments

    # Initialize an empty tensor for the aggregated outputs
    aggregated = torch.zeros((num_batches, num_classes), device=scores.device)

    # Using slice assignment for clarity and a different loop variable
    for batch_index in range(num_batches):
        segment_scores = scores[batch_index * segments : (batch_index + 1) * segments]

        if method == 'mean':
            # Compute the mean across the specified axis directly in the assignment
            aggregated[batch_index] = segment_scores.mean(dim=0)
        else:
            # Using .max() directly without the additional indexing used previously
            aggregated[batch_index], _ = segment_scores.max(dim=0)

    return aggregated


def train_multiview(
        model: nn.Module,
        train_loader: DataLoader, 
        optimizer: torch.optim.Optimizer, 
        loss_fn: nn.Module,
        num_segments: int,
        device: str,
        description: str = "",
        method: str = "mean"
    ) -> None:
    """
    Trains the given model using the provided data loader, optimizer, and loss function.

    Args:
        model (nn.Module): The neural network model to be trained.
        train_loader (DataLoader): The data loader containing the training dataset.
        optimizer (torch.optim.Optimizer): The optimizer used for updating model parameters.
        loss_fn (nn.Module): The loss function used to compute the training loss.
        device (str): The device on which the model and data should be processed ('cuda' or 'cpu').
        description (str, optional): Additional information for tracking epoch description during training. Defaults to "".

    Returns:
        None
    """
    model.train()
    pbar = tqdm(train_loader, desc=description, total=len(train_loader))
    loss_train_mean = statistics.RollingMean(window_size=len(train_loader))
    hits = count = 0 # auxiliary variables for computing accuracy
    for batch in pbar:
        # Gather batch and move to device
        clips, labels = batch['clips'].to(device), batch['labels'].to(device)
        # Forward pass
        outputs = model(clips)
        scores = torch.nn.functional.log_softmax(outputs, dim=-1)
        # Agregate outputs per video
        aggregated_outputs = aggregate_scores(scores, num_segments, method=method)
        # Compute loss
        loss = loss_fn(aggregated_outputs, labels)
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # Update progress bar with metrics
        loss_iter = loss.item()
        hits_iter = torch.eq(aggregated_outputs.argmax(dim=1), labels).sum().item()
        hits += hits_iter
        count += len(labels)
        pbar.set_postfix(
            loss=loss_iter,
            loss_mean=loss_train_mean(loss_iter),
            acc=(float(hits_iter) / len(labels)),
            acc_mean=(float(hits) / count)
        )

    return loss_train_mean(loss_iter), float(hits) / count

def create_datasets_multiview(
        frames_dir: str,
        annotations_dir: str,
        split: TSNDataset.Split,
        clip_length: int,
        crop_size: int,
        temporal_stride: int,
        num_segments: int
) -> Dict[str, TSNDataset]:
    """
    Creates datasets for training, validation, and testing.

    Args:
        frames_dir (str): Directory containing the video frames (a separate directory per video).
        annotations_dir (str): Directory containing annotation files.
        split (TSNDataset.Split): Dataset split (TEST_ON_SPLIT_1, TEST_ON_SPLIT_2, TEST_ON_SPLIT_3).
        clip_length (int): Number of frames of the clips.
        crop_size (int): Size of spatial crops (squares).
        temporal_stride (int): Receptive field of the model will be (clip_length * temporal_stride) / FPS.
        num_clips (int): Number of clips for inference.
        five_crop_size (int): Crop size used in FiveCrop.

    Returns:
        Dict[str, TSNDataset]: A dictionary containing the datasets for training, validation, and testing.
    """
    datasets = {}
    for regime in TSNDataset.Regime:
        datasets[regime.name.lower()] = TSNDataset(
            frames_dir,
            annotations_dir,
            split,
            regime,
            clip_length,
            crop_size,
            temporal_stride,
            num_segments
        )
    
    return datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a video classification model on HMDB51 dataset.')
    parser.add_argument('frames_dir', type=str, 
                        help='Directory containing video files')
    parser.add_argument('--annotations-dir', type=str, default="data/hmdb51/testTrainMulti_601030_splits",
                        help='Directory containing annotation files')
    parser.add_argument('--clip-length', type=int, default=4,
                        help='Number of frames of the clips')
    parser.add_argument('--crop-size', type=int, default=182,
                        help='Size of spatial crops (squares)')
    parser.add_argument('--temporal-stride', type=int, default=6,
                        help='Receptive field of the model will be (clip_length * temporal_stride) / FPS')
    parser.add_argument('--model-name', type=str, default='x3d_xs',
                        help='Model name as defined in models/model_creator.py')
    parser.add_argument('--load-pretrain', action='store_true', default=False,
                        help='Load pretrained weights for the model (if available)')
    parser.add_argument('--optimizer-name', type=str, default="adam",
                        help='Optimizer name (supported: "adam" and "sgd" for now)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for the training data loader')
    parser.add_argument('--batch-size-eval', type=int, default=16,
                        help='Batch size for the evaluation data loader')
    parser.add_argument('--validate-every', type=int, default=5,
                        help='Number of epochs after which to validate the model')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of worker processes for data loading')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training (cuda or cpu)')
    parser.add_argument('--model-path', type=str, default="./weights/weights_multiview_task4_1.pth",
                        help="Path from where to load the model or save (if training)")
    parser.add_argument('--mode', type=str, default="train", choices=["train", "evaluate"],
                        help="Mode for running the script, either train or evaluate")
    parser.add_argument('--num-segments', type=int, default=3,
                        help='Number of segments to divide the video into')

    args = parser.parse_args()
    wandb.init(project="C6_w5_t41", config=args)

    # Create datasets
    datasets = create_datasets_multiview(
        frames_dir=args.frames_dir,
        annotations_dir=args.annotations_dir,
        split=TSNDataset.Split.TEST_ON_SPLIT_1, # hardcoded
        clip_length=args.clip_length,
        crop_size=args.crop_size,
        temporal_stride=args.temporal_stride,
        num_segments=args.num_segments,
    )
    
    # Create data loaders
    loaders = create_dataloaders(
        datasets,
        args.batch_size,
        batch_size_eval=args.batch_size_eval,
        num_workers=args.num_workers
    )

    if args.mode == "train":
        # Init model, optimizer, and loss function
        model = model_creator.create(args.model_name, args.load_pretrain, datasets["training"].get_num_classes())
        
        wandb.watch(model)
        
        optimizer = create_optimizer(args.optimizer_name, model.parameters(), lr=args.lr)
        loss_fn = nn.CrossEntropyLoss()

        model = model.to(args.device)

        early_stopping = EarlyStopping(patience=5, min_delta=0.1) 

        for epoch in range(args.epochs):
            # Validation
            if epoch % args.validate_every == 0:
                description = f"Validation [Epoch: {epoch+1}/{args.epochs}]"
                val_loss, val_accuracy = evaluate(model, loaders['validation'], loss_fn, args.device, description=description)
                early_stopping(val_loss)
                
                wandb.log({"Validation Loss": val_loss, "Validation Accuracy": val_accuracy})

                if early_stopping.early_stop:
                    print(f"Early stopping in epoch {epoch+1}")
                    break

            # Training
            description = f"Training [Epoch: {epoch+1}/{args.epochs}]"
            train_loss, train_accuracy = train_multiview(model, loaders['training'], optimizer, loss_fn, args.num_segments, args.device, description=description)
            wandb.log({"Training Loss": train_loss, "Training Accuracy": train_accuracy})
        
        if args.model_path:
            torch.save(model.state_dict(), args.model_path)

    else:
        model = model_creator.load_model(
            args.model_name, 
            args.load_pretrain, 
            datasets["training"].get_num_classes(),
            args.model_path
        )
        
        model = model.to(args.device)
        loss_fn = nn.CrossEntropyLoss()
    
    # Testing
    evaluate(model, loaders['validation'], loss_fn, args.device, description=f"Validation [Final]")
    evaluate(model, loaders['testing'], loss_fn, args.device, description=f"Testing")
    
    wandb.finish()

    exit()
