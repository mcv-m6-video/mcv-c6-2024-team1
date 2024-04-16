""" Main script for training a video classification model on HMDB51 dataset. """

import argparse
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.HMDB51Dataset import HMDB51Dataset
from datasets.HMDB51DatasetSpatial import HMDB51DatasetSpatial
from models import model_creator
from train import (create_dataloaders, create_datasets, create_optimizer,
                   evaluate, print_model_summary, train)
from utils import statistics
from utils.early_stopping import EarlyStopping


def evaluate_multiview(
        model: nn.Module, 
        valid_loader: DataLoader, 
        loss_fn: nn.Module,
        device: str,
        description: str = ""
    ) -> None:
    """
    Evaluates the given model using the provided data loader and loss function.

    Args:
        model (nn.Module): The neural network model to be validated.
        valid_loader (DataLoader): The data loader containing the validation dataset.
        loss_fn (nn.Module): The loss function used to compute the validation loss (not used for backpropagation)
        device (str): The device on which the model and data should be processed ('cuda' or 'cpu').
        description (str, optional): Additional information for tracking epoch description during training. Defaults to "".

    Returns:
        None
    """
    model.eval()
    pbar = tqdm(valid_loader, desc=description, total=len(valid_loader))
    loss_valid_mean = statistics.RollingMean(window_size=len(valid_loader))
    hits = count = 0 # auxiliary variables for computing accuracy
    for batch in pbar:
        # Gather batch and move to device
        clips, labels = batch['clips'].to(device), batch['labels'].to(device)
        # Forward pass 
        with torch.no_grad():
            outputs = []
            losses = []
            # clips shape (B, C, 5, T, H, W)
            for i in range(clips.size(1)):  # loop over clips per video
                for j in range(clips.size(2)):  # loop over clips per video
                    curr_clip = clips[:, i, j]  # clip is now (B, C, T, H, W)
                    curr_outputs = model(curr_clip)
                    curr_loss = loss_fn(curr_outputs, labels) 
                    curr_loss_iter = curr_loss.item()
                    outputs.append(curr_outputs.cpu())
                    losses.append(curr_loss_iter)
            
            outputs = np.array(outputs)
            agg_outputs = torch.mean(torch.tensor(outputs, device=device), dim=0)
            agg_hits_iter = torch.eq(agg_outputs.argmax(dim=1), labels).sum().item()
            agg_losses = np.mean(losses)
            count += len(labels)
            hits += agg_hits_iter

            # Update progress bar with metrics
            pbar.set_postfix(
                loss=agg_losses,
                loss_mean=loss_valid_mean(agg_losses),
                acc=(float(agg_hits_iter) / len(labels)),
                acc_mean=(float(hits) / count)
            )


def create_datasets_multiview(
        frames_dir: str,
        annotations_dir: str,
        split: HMDB51Dataset.Split,
        clip_length: int,
        crop_size: int,
        temporal_stride: int,
        num_clips: int,
        five_crop_size: int
) -> Dict[str, HMDB51DatasetSpatial]:
    """
    Creates datasets for training, validation, and testing.

    Args:
        frames_dir (str): Directory containing the video frames (a separate directory per video).
        annotations_dir (str): Directory containing annotation files.
        split (HMDB51Dataset.Split): Dataset split (TEST_ON_SPLIT_1, TEST_ON_SPLIT_2, TEST_ON_SPLIT_3).
        clip_length (int): Number of frames of the clips.
        crop_size (int): Size of spatial crops (squares).
        temporal_stride (int): Receptive field of the model will be (clip_length * temporal_stride) / FPS.
        num_clips (int): Number of clips for inference.
        five_crop_size (int): Crop size used in FiveCrop.
    Returns:
        Dict[str, HMDB51DatasetSpatial]: A dictionary containing the datasets for training, validation, and testing.
    """
    datasets = {}
    datasets["validation"] = HMDB51DatasetSpatial(
        frames_dir,
        annotations_dir,
        split,
        HMDB51DatasetSpatial.Regime.VALIDATION,
        clip_length,
        crop_size,
        temporal_stride,
        num_clips=num_clips,
        five_crop_size=five_crop_size
    )
    datasets["testing"] = HMDB51DatasetSpatial(
        frames_dir,
        annotations_dir,
        split,
        HMDB51DatasetSpatial.Regime.TESTING,
        clip_length,
        crop_size,
        temporal_stride,
        num_clips=num_clips,
        five_crop_size=five_crop_size
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
    parser.add_argument('--model-path', type=str, default="./weights/weights_multiview_inference.pth",
                        help="Path from where to load the model or save (if training)")
    parser.add_argument('--mode', type=str, default="evaluate", choices=["train", "evaluate"],
                        help="Mode for running the script, either train or evaluate")
    parser.add_argument('--num-clips', type=int, default=3,
                        help='Number of clips to divide the video into')
    parser.add_argument('--five-crop-size', type=int, default=91,
                        help='Size value used in FiveCrop')

    args = parser.parse_args()

    # Create datasets
    datasets = create_datasets(
        frames_dir=args.frames_dir,
        annotations_dir=args.annotations_dir,
        split=HMDB51Dataset.Split.TEST_ON_SPLIT_1, # hardcoded
        clip_length=args.clip_length,
        crop_size=args.crop_size,
        temporal_stride=args.temporal_stride
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
        optimizer = create_optimizer(args.optimizer_name, model.parameters(), lr=args.lr)
        loss_fn = nn.CrossEntropyLoss()

        print_model_summary(model, args.clip_length, args.crop_size)

        model = model.to(args.device)

        early_stopping = EarlyStopping(patience=5, min_delta=0.15) 

        for epoch in range(args.epochs):
            # Validation
            if epoch % args.validate_every == 0:
                description = f"Validation [Epoch: {epoch+1}/{args.epochs}]"
                val_loss = evaluate(model, loaders['validation'], loss_fn, args.device, description=description)
                early_stopping(val_loss)

                if early_stopping.early_stop:
                    print(f"Early stopping in epoch {epoch+1}")
                    break

            # Training
            description = f"Training [Epoch: {epoch+1}/{args.epochs}]"
            train(model, loaders['training'], optimizer, loss_fn, args.device, description=description, save_path=args.model_path)
    else:
        model = model_creator.load_model(
            args.model_name, 
            args.load_pretrain, 
            datasets["training"].get_num_classes(),
            args.model_path
        )
        
        model = model.to(args.device)
        loss_fn = nn.CrossEntropyLoss()

    datasets = create_datasets_multiview(
        frames_dir=args.frames_dir,
        annotations_dir=args.annotations_dir,
        split=HMDB51DatasetSpatial.Split.TEST_ON_SPLIT_1, # hardcoded
        clip_length=args.clip_length,
        crop_size=args.crop_size,
        temporal_stride=args.temporal_stride,
        num_clips=args.num_clips,
        five_crop_size=args.five_crop_size
    )

    loaders = create_dataloaders(
        datasets,
        args.batch_size,
        batch_size_eval=args.batch_size_eval,
        num_workers=args.num_workers
    )
    
    # Testing
    evaluate_multiview(model, loaders['validation'], loss_fn, args.device, description=f"Validation [Final]")
    evaluate_multiview(model, loaders['testing'], loss_fn, args.device, description=f"Testing")

    exit()
