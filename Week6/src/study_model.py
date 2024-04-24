""" Main script for training a video classification model on HMDB51 dataset. """

import argparse
from models import model_creator
from utils import model_analysis
import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datasets.HMDB51Dataset import HMDB51Dataset

CLASS_NAMES = [
        "brush_hair", "catch", "clap", "climb_stairs", "draw_sword", "drink", 
        "fall_floor", "flic_flac", "handstand", "hug", "kick", "kiss", "pick", 
        "pullup", "push", "ride_bike", "run", "shoot_ball", "shoot_gun", "situp", 
        "smoke", "stand", "sword", "talk", "turn", "wave", 
        "cartwheel", "chew", "climb", "dive", "dribble", "eat", "fencing", 
        "golf", "hit", "jump", "kick_ball", "laugh", "pour", "punch", "pushup", 
        "ride_horse", "shake_hands", "shoot_bow", "sit", "smile", "somersault", 
        "swing_baseball", "sword_exercise", "throw", "walk"
    ]
def generate_per_class_accuracy_plot(per_class_accuracy, name):
        """
        Generates a bar plot of per-class accuracies.

        Args:
            per_class_accuracy (list): List of per-class accuracies.

        Returns:
            None
        """
        # Plot per-class accuracy
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(per_class_accuracy)), per_class_accuracy)
        plt.xlabel('Class')
        plt.ylabel('Accuracy')
        plt.title('Per-Class Accuracy')
        plt.xticks(range(len(per_class_accuracy)), [f"{CLASS_NAMES[i]}" for i in range(len(per_class_accuracy))], rotation=45, ha='right')
        plt.tight_layout()

        # Create directory if it doesn't exist
        output_dir = '/ghome/group01/mcv-c6-2024-team1/Week6/plots/'
        os.makedirs(output_dir, exist_ok=True)

        # Save the plot as an image
        plt.savefig(os.path.join(output_dir, f'{name}.png'))
        
def evaluate_per_class_accuracy(
        model: nn.Module, 
        valid_loader: DataLoader, 
        device: str,
        description: str = ""
    ) -> np.array:
    """
    Evaluates the per-class accuracy of the given model using the provided data loader.

    Args:
        model (nn.Module): The neural network model to be validated.
        valid_loader (DataLoader): The data loader containing the validation dataset.
        device (str): The device on which the model and data should be processed ('cuda' or 'cpu').
        description (str, optional): Additional information for tracking epoch description during training. Defaults to "".

    Returns:
        np.array: Array of per-class accuracies.
    """
    model.eval()
    pbar = tqdm(valid_loader, desc=description, total=len(valid_loader))
    class_correct = np.zeros(valid_loader.dataset.get_num_classes())
    class_total = np.zeros(valid_loader.dataset.get_num_classes())
    for batch in pbar:
        clips, labels = batch['clips'].to(device), batch['labels'].to(device)
        with torch.no_grad():
            outputs = model(clips)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    per_class_accuracy = class_correct / class_total
    return per_class_accuracy
def create_datasets(
        frames_dir: str,
        annotations_dir: str,
        split: HMDB51Dataset.Split,
        clip_length: int,
        crop_size: int,
        temporal_stride: int
) -> Dict[str, HMDB51Dataset]:
    """
    Creates datasets for training, validation, and testing.

    Args:
        frames_dir (str): Directory containing the video frames (a separate directory per video).
        annotations_dir (str): Directory containing annotation files.
        split (HMDB51Dataset.Split): Dataset split (TEST_ON_SPLIT_1, TEST_ON_SPLIT_2, TEST_ON_SPLIT_3).
        clip_length (int): Number of frames of the clips.
        crop_size (int): Size of spatial crops (squares).
        temporal_stride (int): Receptive field of the model will be (clip_length * temporal_stride) / FPS.

    Returns:
        Dict[str, HMDB51Dataset]: A dictionary containing the datasets for training, validation, and testing.
    """
    datasets = {}
    for regime in HMDB51Dataset.Regime:
        datasets[regime.name.lower()] = HMDB51Dataset(
            frames_dir,
            annotations_dir,
            split,
            regime,
            clip_length,
            crop_size,
            temporal_stride
        )
    
    return datasets


def create_dataloaders(
        datasets: Dict[str, HMDB51Dataset],
        batch_size: int,
        batch_size_eval: int = 8,
        num_workers: int = 2,
        pin_memory: bool = True
    ) -> Dict[str, DataLoader]:
    """
    Creates data loaders for training, validation, and testing datasets.

    Args:
        datasets (Dict[str, HMDB51Dataset]): A dictionary containing datasets for training, validation, and testing.
        batch_size (int, optional): Batch size for the data loaders. Defaults to 8.
        num_workers (int, optional): Number of worker processes for data loading. Defaults to 2.
        pin_memory (bool, optional): Whether to pin memory in DataLoader for faster GPU transfer. Defaults to True.

    Returns:
        Dict[str, DataLoader]: A dictionary containing data loaders for training, validation, and testing datasets.
    """
    dataloaders = {}
    for key, dataset in datasets.items():
        dataloaders[key] = DataLoader(
            dataset,
            batch_size=(batch_size if key == 'training' else batch_size_eval),
            shuffle=(key == 'training'),  # Shuffle only for training dataset
            collate_fn=dataset.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
            
    return dataloaders
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a video classification model on HMDB51 dataset.')
    parser.add_argument('--model-name', type=str, default='x3d_xs',
                        help='Model name as defined in models/model_creator.py')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training (cuda or cpu)')
    
    parser.add_argument('--annotations-dir', type=str, default="data/hmdb51/testTrainMulti_601030_splits",
                        help='Directory containing annotation files')
    parser.add_argument('--clip-length', type=int, default=4,
                        help='Number of frames of the clips')
    parser.add_argument('--crop-size', type=int, default=182,
                        help='Size of spatial crops (squares)')
    parser.add_argument('--temporal-stride', type=int, default=12,
                        help='Receptive field of the model will be (clip_length * temporal_stride) / FPS')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for the training data loader')
    parser.add_argument('--batch-size-eval', type=int, default=16,
                        help='Batch size for the evaluation data loader')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of worker processes for data loading')
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
    args = parser.parse_args()
    model = model_creator.create(
        args.model_name,
        True, 
        51
    )

    model = model.to(args.device)
    print("Number parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("GFLOPS:", model_analysis.calculate_operations(model, clip_length=16, crop_height=224, crop_width=224))
    
    per_class_accuracy = evaluate_per_class_accuracy(model, loaders['validation'], args.device)

    # Print per-class accuracy
    print("Per-Class Accuracy:")
    for i, accuracy in enumerate(per_class_accuracy):
        print(f"Class {CLASS_NAMES[i]}: {accuracy:.4f}")
    generate_per_class_accuracy_plot(per_class_accuracy, "lighter_3dx_model_plots")
    exit()