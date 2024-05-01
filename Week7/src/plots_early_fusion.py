import pickle
import argparse
from typing import Dict

import torch
from utils.plots import Plots
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.HMDB51Dataset import HMDB51Dataset
from utils import rolling_mean
from train import create_dataloaders, create_datasets, create_optimizer
from models import model_creator
from models.FusionModalitiesNetwork import FusionModalitiesNet

TEST_EMBEDDINGS_PATH  = "/ghome/group01/mcv-c6-2024-team1/Week7/results/embeddings_posec3d_test.pkl"

def load_test_embeddings(test_path: str) -> Dict:
    with open(test_path, "rb") as f:
        test_embeddings_keypoints = pickle.load(f)

    return test_embeddings_keypoints


def evaluate(
    model: nn.Module,
    valid_loader: DataLoader,
    loss_fn: nn.Module,
    device: str,
    description: str = "",
    test_embeddings_keypoints: Dict = None,
):
    """
    Evaluates the given model using the provided data loader and loss function.

    Args:
        model (nn.Module): The neural network model to be validated.
        valid_loader (DataLoader): The data loader containing the validation dataset.
        loss_fn (nn.Module): The loss function used to compute the validation loss (not used for backpropagation)
        device (str): The device on which the model and data should be processed ('cuda' or 'cpu').
        description (str, optional): Additional information for tracking epoch description during training. Defaults to "".
        test_embeddings_keypoints (Dict): Dictionary containing valid/test embeddings from keypoints.
    Returns:
        None
    """
    model.eval()
    pbar = tqdm(valid_loader, desc=description, total=len(valid_loader))
    loss_valid_mean = rolling_mean.RollingMean(window_size=len(valid_loader))
    hits = count = 0  # auxiliary variables for computing accuracy
    y_pred = []
    y_test = []
    
    for batch in pbar:
        clips, labels, paths = (
            batch["clips"].to(device),
            batch["labels"].to(device),
            batch["paths"],
        )
        paths = [path.split("/")[-1] for path in paths]
        embeddings = torch.FloatTensor([test_embeddings_keypoints[path] for path in paths]).to(device)

        with torch.no_grad():
            outputs = model(clips, embeddings)

            # Compute loss (just for logging, not used for backpropagation)
            loss = loss_fn(outputs, labels)

            # Compute metrics
            loss_iter = loss.item()
            y_pred.extend(outputs.argmax(dim=1).cpu().numpy())
            y_test.extend(labels.cpu().numpy())
            hits_iter = torch.eq(outputs.argmax(dim=1), labels).sum().item()
            hits += hits_iter
            count += len(labels)

            # Update progress bar with metrics
            pbar.set_postfix(
                loss=loss_iter,
                loss_mean=loss_valid_mean(loss_iter),
                acc=(float(hits_iter) / len(labels)),
                acc_mean=(float(hits) / count),
            )

    return y_test, y_pred


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument("frames_dir", type=str, help="Directory containing video files")
    parser.add_argument(
        "--annotations-dir",
        type=str,
        default="data/hmdb51/testTrainMulti_601030_splits",
        help="Directory containing annotation files",
    )
    parser.add_argument(
        "--clip-length", 
        type=int, 
        default=16, 
        help="Number of frames of the clips"
    )
    parser.add_argument(
        "--crop-size", 
        type=int, 
        default=256, 
        help="Size of spatial crops (squares)"
    )
    parser.add_argument(
        "--temporal-stride",
        type=int,
        default=5,
        help="Receptive field of the model will be (clip_length * temporal_stride) / FPS",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="x3d_m",
        help="Model name as defined in models/model_creator.py",
    )
    parser.add_argument(
        "--load-pretrain",
        action="store_true",
        default=True,
        help="Load pretrained weights for the model (if available)",
    )
    parser.add_argument(
        "--optimizer-name",
        type=str,
        default="adam",
        help='Optimizer name (supported: "adam" and "sgd" for now)',
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=1e-4, 
        help="Learning rate"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=50, 
        help="Number of epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for the training data loader",
    )
    parser.add_argument(
        "--batch-size-eval",
        type=int,
        default=16,
        help="Batch size for the evaluation data loader",
    )
    parser.add_argument(
        "--validate-every",
        type=int,
        default=5,
        help="Number of epochs after which to validate the model",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of worker processes for data loading",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training (cuda or cpu)",
    )
    parser.add_argument(
        "--video-model-path",
        type=str,
        default="weights/x3d_m.pth",
        help="Path in which to save weights",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="weights/early_fusion.pth",
        help="Path in which to save weights",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    
    test_embeddings_keypoints = load_test_embeddings(TEST_EMBEDDINGS_PATH)
    datasets = create_datasets(
        frames_dir=args.frames_dir,
        annotations_dir=args.annotations_dir,
        split=HMDB51Dataset.Split.TEST_ON_SPLIT_1,
        clip_length=args.clip_length,
        crop_size=args.crop_size,
        temporal_stride=args.temporal_stride,
    )

    loaders = create_dataloaders(
        datasets,
        args.batch_size,
        batch_size_eval=args.batch_size_eval,
        num_workers=args.num_workers,
    )

    video_model = model_creator.load_video_model(
        args.model_name, 
        args.load_pretrain, 
        datasets["training"].get_num_classes(),
        embedding_size=512,
        load_path=args.video_model_path
    )
    
    video_model = video_model.to(args.device)
    model = FusionModalitiesNet(video_model=video_model).to(args.device)
    optimizer = create_optimizer(args.optimizer_name, model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    state = torch.load(args.model_path)
    model.load_state_dict(state)

    # Testing
    y_test, y_pred = evaluate(
        model, 
        loaders["testing"], 
        loss_fn, 
        args.device, 
        description=f"Testing", 
        test_embeddings_keypoints=test_embeddings_keypoints
    )

    Plots.generate_confusion_matrix(y_test, y_pred, "/ghome/group01/mcv-c6-2024-team1/Week7/plots/confusion_fusion.png")
    Plots.generate_per_class_accuracy_plot(y_test, y_pred, "/ghome/group01/mcv-c6-2024-team1/Week7/plots/per_class_acc_fusion.png")
    
    exit()
