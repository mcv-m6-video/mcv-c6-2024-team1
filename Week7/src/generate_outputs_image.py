""" Main script for training a video classification model on HMDB51 dataset. """
import pickle
import argparse
from collections import defaultdict

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.HMDB51Dataset import HMDB51Dataset
from models import model_creator
from train import create_dataloaders, create_datasets
from utils import statistics

def save_outputs(
    model: nn.Module, 
    valid_loader: DataLoader, 
    loss_fn: nn.Module,
    device: str,
    description: str = "",
    save_file: str = "/ghome/group01/mcv-c6-2024-team1/Week7/results/results_week6.pkl",
    gt_path: str = "/ghome/group01/mcv-c6-2024-team1/Week7/results/gt.pkl"
) -> None:
    """
    Evaluates the given model using the provided data loader and loss function.

    Args:
        model (nn.Module): The neural network model to be validated.
        valid_loader (DataLoader): The data loader containing the validation dataset.
        loss_fn (nn.Module): The loss function used to compute the validation loss (not used for backpropagation)
        device (str): The device on which the model and data should be processed ('cuda' or 'cpu').
        description (str, optional): Additional information for tracking epoch description during training. Defaults to "".
        save_dir (str, optional): The directory to save the evaluation results. Defaults to "/results".
        save_file_prefix (str, optional): The prefix for the output files. Defaults to "model_outputs".

    Returns:
        None
    """
    model.eval()
    save_path = save_file

    pbar = tqdm(valid_loader, desc=description, total=len(valid_loader))
    loss_valid_mean = statistics.RollingMean(window_size=len(valid_loader))

    gt = {}
    outputs_dict = {}
    hits = count = 0  # auxiliary variables for computing accuracy
    i = 0
    for batch in pbar:
        i=i+1
        print("\n-In batch ",i)
        # Gather batch and move to device
        clips, labels, paths = batch['clips'].to(device), batch['labels'].to(device), batch['paths']
        paths = [path.split('/')[-1] for path in paths]

        # Forward pass
        with torch.no_grad():
            outputs = model(clips)
            print("Outputs: ", outputs.argmax(dim=1))
            print("Labels: ", labels)
            hits_iter = torch.eq(outputs.argmax(dim=1), labels).sum().item()
            hits += hits_iter
            count += len(labels)
            
            outputs_cpu = outputs.cpu().numpy()
            outputs_dict.update({paths[i]: outputs_cpu[i] for i in range(len(paths))})
            gt.update({paths[i]: labels[i].cpu().numpy() for i in range(len(paths))})

            # Compute loss (just for logging, not used for backpropagation)
            loss = loss_fn(outputs, labels) 

            # Update progress bar with metrics
            pbar.set_postfix(
                loss=loss.item(),
                loss_mean=loss_valid_mean(loss.item())
            )
            
    accuracy = hits / count
    print("Test Accuracy (%):", accuracy * 100)

    # Save outputs to file
    with open(save_path, 'wb') as f:
        pickle.dump(outputs_dict, f)
        
    with open(gt_path, 'wb') as f:
        pickle.dump(gt, f)

def evaluate_classes(
    model: nn.Module, 
    valid_loader: DataLoader, 
    loss_fn: nn.Module,
    device: str,
    description: str = ""
) -> dict:
    """
    Evaluates the given model using the provided data loader and loss function.

    Args:
        model (nn.Module): The neural network model to be validated.
        valid_loader (DataLoader): The data loader containing the validation dataset.
        loss_fn (nn.Module): The loss function used to compute the validation loss (not used for backpropagation)
        device (str): The device on which the model and data should be processed ('cuda' or 'cpu').
        description (str, optional): Additional information for tracking epoch description during training. Defaults to "".

    Returns:
        dict: A dictionary containing per-class accuracy.
    """
    model.eval()
    pbar = tqdm(valid_loader, desc=description, total=len(valid_loader))
    loss_valid_mean = statistics.RollingMean(window_size=len(valid_loader))
    hits = count = 0  # auxiliary variables for computing accuracy
    per_class_hits = defaultdict(int)
    per_class_count = defaultdict(int)

    for batch in pbar:
        # Gather batch and move to device
        clips, labels = batch['clips'].to(device), batch['labels'].to(device)
        # Forward pass
        with torch.no_grad():
            outputs = model(clips)
            # Compute loss (just for logging, not used for backpropagation)
            loss = loss_fn(outputs, labels) 
            # Compute metrics
            loss_iter = loss.item()
            hits_iter = torch.eq(outputs.argmax(dim=1), labels).sum().item()
            hits += hits_iter
            count += len(labels)
            # Update per-class counts
            for pred, target in zip(outputs.argmax(dim=1), labels):
                per_class_hits[target.item()] += int(pred == target)
                per_class_count[target.item()] += 1
            # Update progress bar with metrics
            pbar.set_postfix(
                loss=loss_iter,
                loss_mean=loss_valid_mean(loss_iter),
                acc=(float(hits_iter) / len(labels)),
                acc_mean=(float(hits) / count)
            )

    accuracy = hits / count
    per_class_accuracy = {cls: per_class_hits[cls] / per_class_count[cls] if per_class_count[cls] != 0 else 0
                          for cls in per_class_hits}
    
    print("Accuracy: " + str(accuracy))
    return per_class_accuracy


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
    parser.add_argument('--model-name', type=str, default='x3d_m',
                        help='Model name as defined in models/model_creator.py')
    parser.add_argument('--load-pretrain', action='store_true', default=False,
                    help='Load pretrained weights for the model (if available)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for the training data loader')
    parser.add_argument('--batch-size-eval', type=int, default=16,
                        help='Batch size for the evaluation data loader')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of worker processes for data loading')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training (cuda or cpu)')
    parser.add_argument('--model-path', type=str, default='./weights/model_rgb_dropout.pth',
                        help='Path to load the model from')

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

    model = model_creator.load_model(
        args.model_name, 
        args.load_pretrain, 
        datasets["training"].get_num_classes(),
        load_path=args.model_path
    )
    
    model = model.to(args.device)
    loss_fn = nn.CrossEntropyLoss()

    # Testing
    results_dir = "/ghome/group01/mcv-c6-2024-team1/Week7/results/results_x3d.pkl"

    # Compute results and save them if directories don't exist
    print("Computing results...")
    save_outputs(model, loaders['testing'], loss_fn, args.device, description=f"Testing", save_file=results_dir)
