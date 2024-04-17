""" Main script for training a video classification model on HMDB51 dataset. """

import argparse
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.HMDB51Dataset import HMDB51Dataset

import wandb
from datasets.TSNDataset_improvement2 import TSNDatasetImprov
from models import model_creator
from train import create_dataloaders, create_optimizer, evaluate, print_model_summary, create_datasets
from utils import statistics
from utils.early_stopping import EarlyStopping


def train_multiview(
        model: nn.Module,
        train_loader: DataLoader, 
        optimizer: torch.optim.Optimizer, 
        loss_fn: nn.Module,
        device: str,
        description: str = "",
        save_path: str = None
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
        
        #####
        # Gather batch and move to device
        clips, labels = batch['clips'].to(device), batch['labels'].to(device)
        
        batch_size, num_clips, channels, clip_length, height, width = clips.shape
        # Forward pass
        optimizer.zero_grad()  # Zero the gradients
        #outputs = torch.zeros((num_clips, batch_size, 51), requires_grad=True).to(device)
        outputs = []
        losses = []
        # clips shape (B, 3, C, T, H, W)
        for j in range(clips.size(1)):  # loop over clips per video
            output = model(clips[:, j])
            outputs.append(output)
            #outputs[j] = output # Predict every clip per separate
        outputs = torch.stack(outputs)
        agg_outputs = torch.mean(outputs, dim=0) # Aggregate predictions
        print("Outputs shape: ", agg_outputs.shape)
        print("Labels shape: ", labels.shape)
        loss = loss_fn(agg_outputs, labels) # Loss takes into account all snippets and aggregation function
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # Update progress bar with metrics
        loss_iter = loss.item()
        hits_iter = torch.eq(agg_outputs.argmax(dim=1), labels).sum().item()
        hits += hits_iter
        count += len(labels)
        pbar.set_postfix(
            loss=loss_iter,
            loss_mean=loss_train_mean(loss_iter),
            acc=(float(hits_iter) / len(labels)),
            acc_mean=(float(hits) / count)
        )
        
    if save_path:
        torch.save(model.state_dict(), save_path)
    
    return sum(loss_train_mean.data) / len(loss_train_mean.data), hits / count

# TODO: REMOVE
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
            # clips shape (B, C, 5, t, T, H, W)
            for i in range(clips.size(1)):  # loop over clips per video
                for j in range(clips.size(2)):  # loop over clips per video
                    curr_clip = clips[:, i, j]  # clip is now (B, C, T, H, W)
                    print(f'curr_clip shape: {curr_clip.shape}')
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
        split: TSNDatasetImprov.Split,
        clip_length: int,
        crop_size: int,
        num_clips: int,
        temporal_stride: int
) -> Dict[str, TSNDatasetImprov]:
    """
    Creates datasets for training, validation, and testing.

    Args:
        frames_dir (str): Directory containing the video frames (a separate directory per video).
        annotations_dir (str): Directory containing annotation files.
        split (TSNDatasetImprov.Split): Dataset split (TEST_ON_SPLIT_1, TEST_ON_SPLIT_2, TEST_ON_SPLIT_3).
        clip_length (int): Number of frames of the clips.
        crop_size (int): Size of spatial crops (squares).
        temporal_stride (int): Receptive field of the model will be (clip_length * temporal_stride) / FPS.
        num_clips (int): Number of clips for inference.
        five_crop_size (int): Crop size used in FiveCrop.

    Returns:
        Dict[str, TSNDatasetImprov]: A dictionary containing the datasets for training, validation, and testing.
    """
    datasets = {}
    for regime in TSNDatasetImprov.Regime:
        datasets[regime.name.lower()] = TSNDatasetImprov(
            frames_dir,
            annotations_dir,
            split,
            regime,
            clip_length,
            crop_size,
            num_clips=num_clips,
            temporal_stride=temporal_stride
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
    parser.add_argument('--mode', type=str, default="train", choices=["train", "evaluate"],
                        help="Mode for running the script, either train or evaluate")
    parser.add_argument('--num-clips', type=int, default=3,
                        help='Number of clips to divide the video into')
    parser.add_argument('--five-crop-size', type=int, default=91,
                        help='Size value used in FiveCrop')

    args = parser.parse_args()
    wandb.init(project="C6_w5_t42", config=args)

    # Create datasets
    datasets_train = create_datasets_multiview(
        frames_dir=args.frames_dir,
        annotations_dir=args.annotations_dir,
        split=TSNDatasetImprov.Split.TEST_ON_SPLIT_1, # hardcoded
        clip_length=args.clip_length,
        crop_size=args.crop_size,
        num_clips=args.num_clips,
        temporal_stride=args.temporal_stride
    )
    
    # Create datasets
    datasets_validation = create_datasets(
        frames_dir=args.frames_dir,
        annotations_dir=args.annotations_dir,
        split=HMDB51Dataset.Split.TEST_ON_SPLIT_1, # hardcoded
        clip_length=args.clip_length,
        crop_size=args.crop_size,
        temporal_stride=args.temporal_stride
    )

    # Create data loaders
    loaders_training = create_dataloaders(
        datasets_train,
        args.batch_size,
        batch_size_eval=args.batch_size_eval,
        num_workers=args.num_workers
    )
    loaders_validation = create_dataloaders(
        datasets_validation,
        args.batch_size,
        batch_size_eval=args.batch_size_eval,
        num_workers=args.num_workers
    )

    if args.mode == "train":
        # Init model, optimizer, and loss function
        model = model_creator.create(args.model_name, args.load_pretrain, datasets_train["training"].get_num_classes())
        
        wandb.watch(model)
        
        optimizer = create_optimizer(args.optimizer_name, model.parameters(), lr=args.lr)
        loss_fn = nn.CrossEntropyLoss()

        print_model_summary(model, args.clip_length, args.crop_size)

        model = model.to(args.device)

        early_stopping = EarlyStopping(patience=5, min_delta=0.15) 

        for epoch in range(args.epochs):
            # Validation
            if epoch % args.validate_every == 0:
                description = f"Validation [Epoch: {epoch+1}/{args.epochs}]"
                val_loss, val_accuracy = evaluate(model, loaders_validation['validation'], loss_fn, args.device, description=description)
                early_stopping(val_loss)
                
                wandb.log({"Validation Loss": val_loss, "Validation Accuracy": val_accuracy})

                if early_stopping.early_stop:
                    print(f"Early stopping in epoch {epoch+1}")
                    break

            # Training
            description = f"Training [Epoch: {epoch+1}/{args.epochs}]"
            train_loss, train_accuracy = train_multiview(model, loaders_training['training'], optimizer, loss_fn, args.device, description=description, save_path=args.model_path)
            wandb.log({"Training Loss": train_loss, "Training Accuracy": train_accuracy})
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
    evaluate(model, loaders_validation['validation'], loss_fn, args.device, description=f"Validation [Final]")
    evaluate(model, loaders_validation['testing'], loss_fn, args.device, description=f"Testing")
    
    wandb.finish()

    exit()
