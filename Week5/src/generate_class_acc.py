""" Main script for training a video classification model on HMDB51 dataset. """

import argparse

import torch.nn as nn

from datasets.HMDB51Dataset import HMDB51Dataset
from models import model_creator
from train import (create_dataloaders, create_datasets, create_optimizer,
                   evaluate, evaluate_per_class_accuracy, print_model_summary,
                   train)
from utils.plots import Plots
from utils.statistics import evaluate_per_class_accuracy

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
    parser.add_argument('--temporal-stride', type=int, default=12,
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

    # Init model, optimizer, and loss function
    model = model_creator.create(args.model_name, args.load_pretrain, datasets["training"].get_num_classes())
    optimizer = create_optimizer(args.optimizer_name, model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    print_model_summary(model, args.clip_length, args.crop_size)

    model = model.to(args.device)

    for epoch in range(args.epochs):
        # Validation
        if epoch % args.validate_every == 0:
            description = f"Validation [Epoch: {epoch+1}/{args.epochs}]"
            evaluate(model, loaders['validation'], loss_fn, args.device, description=description)
        # Training
        description = f"Training [Epoch: {epoch+1}/{args.epochs}]"
        train(model, loaders['training'], optimizer, loss_fn, args.device, description=description)

    # Testing
    evaluate(model, loaders['validation'], loss_fn, args.device, description=f"Validation [Final]")
    evaluate(model, loaders['testing'], loss_fn, args.device, description=f"Testing")
    
    # Compute per-class accuracy
    per_class_accuracy = evaluate_per_class_accuracy(model, loaders['validation'], args.device)

    # Print per-class accuracy
    # print("Per-Class Accuracy:")
    # for i, accuracy in enumerate(per_class_accuracy):
    #     print(f"Class {CLASS_NAMES[i]}: {accuracy:.4f}")
    Plots.generate_per_class_accuracy_plot(per_class_accuracy, "first_model_plots")
    exit()