
import pickle
import argparse
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.HMDB51Dataset import HMDB51Dataset
from train import create_dataloaders, create_datasets, create_optimizer
from models import model_creator
from models.FusionModalitiesNetwork import FusionModalitiesNet
from utils.model_analysis import *



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
    model.load_state_dict(torch.load(args.model_path))

    ops = calculate_operations_fusion(model, args.clip_length, args.crop_size, args.crop_size)
    params = calculate_parameters(model)
    print("Operations of fusion model: ",ops)
    print("Parameters of fusion model: ", params)
    

    exit()
