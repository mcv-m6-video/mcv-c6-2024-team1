from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.HMDB51Dataset import HMDB51Dataset
from datasets.HMDB51DatasetSpatial import HMDB51DatasetSpatial
from models import model_creator
from train import create_dataloaders, create_datasets
from utils import statistics


def optuna_evaluate_multiview(
        model: nn.Module, 
        valid_loader: DataLoader, 
        loss_fn: nn.Module,
        device: str,
        used_crops: list = [],
        description: str = "",
    ) -> float:
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
    accu_mean = 0
    for batch in pbar:
        # Gather batch and move to device
        clips, labels = batch['clips'].to(device), batch['labels'].to(device)
        # Forward pass 
        with torch.no_grad():
            outputs = []
            losses = []
            for i in range(clips.size(1)):  # loop over clips per video
                for j in range(clips.size(2)):  # loop over clips per video
                    if used_crops[j]:
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
            accu_mean = (float(hits) / count)
    return accu_mean

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


def optuna_function(num_clips,crop_1,crop_2,crop_3,crop_4,crop_5):

    used_crops = [crop_1,crop_2,crop_3,crop_4,crop_5]
    # Create datasets
    datasets = create_datasets(
        frames_dir='./frames',
        annotations_dir='data/hmdb51/testTrainMulti_601030_splits',
        split=HMDB51Dataset.Split.TEST_ON_SPLIT_1, # hardcoded
        clip_length=4,
        crop_size=182,
        temporal_stride=2
    )

    # Create data loaders
    loaders = create_dataloaders(
        datasets,
        16,
        batch_size_eval=16,
        num_workers=2
    )

    model = model_creator.load_model(
        'x3d_xs', 
        False, 
        datasets["training"].get_num_classes(),
        './weights/weights_multiview_inference_frozen.pth'
    )
    
    model = model.to('cuda')
    loss_fn = nn.CrossEntropyLoss()

    datasets = create_datasets_multiview(
        frames_dir='./frames',
        annotations_dir='data/hmdb51/testTrainMulti_601030_splits',
        split=HMDB51DatasetSpatial.Split.TEST_ON_SPLIT_1, # hardcoded
        clip_length=4,
        crop_size=182,
        temporal_stride=2,
        num_clips=num_clips,
        five_crop_size=150
    )

    loaders = create_dataloaders(
        datasets,
        16,
        batch_size_eval=16,
        num_workers=2
    )
    
    # Testing
    # optuna_evaluate_multiview(model, loaders['validation'], loss_fn, 'cuda', used_crops,description=f"Validation [Final]")
    acc = optuna_evaluate_multiview(model, loaders['testing'], loss_fn, 'cuda', used_crops,description=f"Testing")

    return acc
