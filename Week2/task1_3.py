import sys
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
import os
from typing import Iterator, List
from task1_1 import run_inference
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from functional import seq
from tqdm import tqdm
import cv2
import json
import numpy as np
from nn import DetectionTransform
from nn.yolo.utils import utils
from nn.yolo_dataset import YoloDataset
from metrics import evaluate, mAP
from week_utils import readXMLtoAnnotation
REPO_DIR = "ultralytics/yolov5"
MODEL_NAME = "yolov5s"
VIDEO_PATH = "../Data/AICity_data_S03_C010/AICity_data/train/S03/c010/vdo.avi"
NUM_CLASSES = 80  # Assuming your new dataset has 80 classes
ANNOTATIONS_PATH = "../Data/AICity_data_S03_C010/ai_challenge_s03_c010-full_annotation.xml"


class Video(Dataset):

    def __getitem__(self, index) -> Image:
        return default_loader(self.frames[index])

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.frames = self.read_video()

    def read_video(self) -> List[Image.Image]:
        frames = []
        cap = cv2.VideoCapture(self.video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Convert frame to Image
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
                frames.append(image)
            else:
                break
        cap.release()
        return frames

    def get_frames(self, start: int = 0, end: int = 2141) -> Iterator[Image.Image]:
        return iter(self.frames[start:end])

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.frames)

    def __str__(self):
        return 'Video(path={})'.format(self.video_path)


def modify_model(model, num_layers_to_freeze=2):
    # Count the number of layers
    num_layers = sum(1 for _ in model.model.children())
    # Determine the index from which to freeze
    start_freeze_index = max(0, num_layers - num_layers_to_freeze)

    # Freeze layers up to the specified index
    layer_index = 0
    last_linear_layer = None
    for param in model.parameters():
        last_linear_layer = last_linear_layer
        if layer_index < start_freeze_index:
            param.requires_grad = False
        layer_index += 1


    # Insert new layers before the last layer
    new_layers = [
        nn.Linear(num_layers, 1024),
        nn.ReLU(inplace=True),  # Add ReLU activation
        nn.Linear(1024, 512),
        nn.ReLU(inplace=True),  # Add ReLU activation
        nn.Linear(512, NUM_CLASSES)  # Adjust for the number of classes
    ]

    # Replace the last linear layer with an identity layer
    for name, child in model.named_children():
        if isinstance(child, nn.Module) and hasattr(child, '__contains__'):
            if last_linear_layer in child:
                for n, module in child.named_children():
                    if module == last_linear_layer:
                        setattr(child, n, nn.Identity())

    # Add new layers before the last linear layer
    for layer in reversed(new_layers):
        model.model.add_module(f'new_layer_{num_layers}', layer)


    # Set requires_grad to True for new layers
    for param in model.parameters():
        if param.requires_grad:
            break  # Stop when we reach the first trainable parameter
        param.requires_grad = True

    return model


def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        print('Starts epoch ',epoch)
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}")

def evaluate_model(model, predictions, annotations):
    # Perform evaluation
    mIoU, precision, recall, f1_score = evaluate(predictions, annotations)
    mAP_val = mAP(annotations, predictions)
    print(f"mIoU: {mIoU}, Precision: {precision}, Recall: {recall}, F1-score: {f1_score}, mAP: {mAP_val}")

def run_training_and_evaluation(train=False, evaluate_only=False):
    # Load YOLOv5s model for fine-tuning
    model = torch.hub.load(REPO_DIR, MODEL_NAME, pretrained=True)
    # Modify YOLOv5 model architecture
    model = modify_model(model)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    if train or not evaluate_only:
        # Fine-tuning YOLOv5 model
        video = Video(VIDEO_PATH)
        detection_transform = DetectionTransform()
        classes = utils.load_classes('../config/coco.names')
        #hyperparams = parse_model_config('../config/yolov5s.cfg')[0]
        gt = readXMLtoAnnotation(ANNOTATIONS_PATH, remParked=False)
        dataset = YoloDataset(video, gt, classes, transforms=detection_transform)
        data_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

        train_model(model, data_loader, criterion, optimizer, num_epochs=10)
        # Save the fine-tuned model
        torch.save(model.state_dict(), "models/fine_tuned_yolov5s.pth")
        run_inference(model=model, name_model="fine_tuned_yolov5s_1")
    if evaluate_only:
        # Load predictions and annotations
        predictions = json.load(open("results/fine_tuned_yolov5s_bbxs_clean_formatted.json", "r"))
        annotations = readXMLtoAnnotation(ANNOTATIONS_PATH, remParked=False)

        # Evaluate the model
        evaluate_model(model, predictions, annotations)

if __name__ == "__main__":
    args = sys.argv[1:]
    if "train" in args:
        run_training_and_evaluation(train=True)
    elif "evaluate" in args:
        run_training_and_evaluation(evaluate_only=True)
    else:
        run_training_and_evaluation()
