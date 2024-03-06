import os

import torch
import torchvision
from sklearn.model_selection import KFold
from tqdm import tqdm

from datasets import create_dataloaders
from metrics import mAP
from week_utils import (convertAnnotations, readXMLtoAnnotation,
                        split_strategy_A,
                        split_video, split_strategy_B, split_strategy_C)

NUM_EPOCHS = 1
LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
UNFROZEN_LAYERS = 3

ANNOTATIONS_PATH = (
    "../Data/AICity_data_S03_C010/ai_challenge_s03_c010-full_annotation.xml"
)
VIDEO_PATH = "../Data/AICity_data_S03_C010/AICity_data/train/S03/c010/vdo.avi"
FRAMES_PATH = "video_frames"
CAR_LABEL = 3

if not os.path.exists(FRAMES_PATH):
    os.makedirs(FRAMES_PATH)
    split_video(VIDEO_PATH)
else:
    print("Frames already extracted")

frames = os.listdir(FRAMES_PATH)


def evaluate(model, test_loader, device, annotations):
    model.eval()
    bbxs = {}

    with torch.no_grad():
        for images, targets in tqdm(test_loader):
            images = [image.to(device) for image in images]
            outputs = model(images)

            for target, output in zip(targets, outputs):
                frame_num = str(target["image_id"].item())
                boxes = output["boxes"]
                labels = output["labels"]

                for i, bbox in enumerate(boxes):
                    x_min, y_min, x_max, y_max = bbox.int().cpu().numpy()

                    if labels[i] == CAR_LABEL:
                        if frame_num not in bbxs:
                            bbxs[frame_num] = []

                        bbxs[frame_num].append(
                            {"bbox": [int(x_min), int(y_min), int(x_max), int(y_max)]}
                        )

    return mAP(annotations, bbxs)


def train(model, train_loader, test_loader, device, annotations):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )

    for epoch in range(NUM_EPOCHS):
        model.train()

        for images, targets in train_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        value_mAP = evaluate(model, test_loader, device, annotations)
        print(f"Epoch: {epoch}, mAP: {value_mAP}, loss: {losses}")
    print("Training finished")

def run_finetuning(device="cpu", strategy="A"):
    annotations = readXMLtoAnnotation(ANNOTATIONS_PATH)
    formatted_annotations = convertAnnotations(annotations)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    transformations = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
        ]
    )
    n_frames = len(frames)

    if strategy == "A":
        train_idxs, test_idxs = split_strategy_A(n_frames)
        train_loader, test_loader = create_dataloaders(
            formatted_annotations,
            VIDEO_PATH,
            train_idxs,
            test_idxs,
            transformations,
            batch_size=2,
        )
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(
            device
        )

        for param_idx, param in enumerate(model.parameters()):
            if param_idx < len(list(model.parameters())) - UNFROZEN_LAYERS:
                param.requires_grad = False

        train(model, train_loader, test_loader, device, annotations)
        map = evaluate(model, test_loader, device, annotations)
        print(
            f"Using NUM_EPOCHS = {NUM_EPOCHS} LEARNING_RATE = {LEARNING_RATE} MOMENTUM = {MOMENTUM} WEIGHT_DECAY = {WEIGHT_DECAY} UNFROZEN_LAYERS ={UNFROZEN_LAYERS} using strategy {strategy}"
        )
        print("Finetuned map: ", map)
    elif strategy == "B" or strategy == "C":
        maps = []
        # K-Fold Cross-Validation
        for fold_idx in range(4):
            if strategy == "B":
                train_idxs, test_idxs = split_strategy_B(fold_idx, n_frames)  # K-Fold split
            else:
                train_idxs, test_idxs = split_strategy_C(n_frames)  # Random split
            print(f"Fold {fold_idx + 1}")
            train_loader, test_loader = create_dataloaders(
                formatted_annotations,
                VIDEO_PATH,
                train_idxs,
                test_idxs,
                transformations,
                batch_size=2,
            )
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)

            for param_idx, param in enumerate(model.parameters()):
                if param_idx < len(list(model.parameters())) - UNFROZEN_LAYERS:
                    param.requires_grad = False

            train(model, train_loader, test_loader, device, annotations)
            map = evaluate(model, test_loader, device, annotations)
            maps.append(map)
            print(
                f"Using NUM_EPOCHS = {NUM_EPOCHS} LEARNING_RATE = {LEARNING_RATE} MOMENTUM = {MOMENTUM} WEIGHT_DECAY = {WEIGHT_DECAY} UNFROZEN_LAYERS ={UNFROZEN_LAYERS} using strategy {strategy}"
            )
            print(f'Fold {fold_idx+1} map: {map}')
        print(f"Average mAP (using strategy {strategy}): {sum(maps) / len(maps)}")
    else:
        raise ValueError("Invalid strategy provided. Please use 'A', 'B', or 'C'.")


if __name__ == "__main__":
    run_finetuning(device="cuda" if torch.cuda.is_available() else "cpu", strategy="A")
