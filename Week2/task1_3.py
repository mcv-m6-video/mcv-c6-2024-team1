import numpy as np

import torch
import torchvision
from week_utils import readXMLtoAnnotation, convertAnnotations
from datasets import create_dataloaders
from tqdm import tqdm
from metrics import mAP

NUM_EPOCHS = 1
LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

FRAME_25_PERCENT = 510
ANNOTATIONS_PATH = (
    "../Data/AICity_data_S03_C010/ai_challenge_s03_c010-full_annotation.xml"
)
VIDEO_PATH = "../Data/AICity_data_S03_C010/AICity_data/train/S03/c010/vdo.avi"

def evaluate(model, test_loader, device, annotations):
    model.eval()
    bbxs = {}
    
    with torch.no_grad():
        for images, targets in tqdm(test_loader):
            images = [image.to(device) for image in images]
            outputs = model(images)
            
            for target, output in zip(targets, outputs):
                frame_num = str(target["image_id"].item())
                boxes = output['boxes']
                labels = output['labels']

                for i, bbox in enumerate(boxes):
                    x_min, y_min, x_max, y_max = bbox.int().cpu().numpy()

                    if labels[i] == 3:
                        if frame_num not in bbxs:
                            bbxs[frame_num] = []

                        bbxs[frame_num].append({"bbox": [int(x_min), int(y_min), int(x_max), int(y_max)]})
            
    return mAP(annotations, bbxs)
    
def train(model, train_loader, test_loader, device, annotations):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    
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
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, mAP: {value_mAP}, loss: {losses}")

def run_finetuning(device="cpu"):
    annotations = readXMLtoAnnotation(ANNOTATIONS_PATH)
    formatted_annotations = convertAnnotations(annotations)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    transformations = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
        ]
    )

    # THIS SHOULD BE CHANGED WHEN K-FOLDING
    train_idxs, test_idxs = np.arange(0, FRAME_25_PERCENT), np.arange(
        FRAME_25_PERCENT, FRAME_25_PERCENT * 4
    )
    train_loader, test_loader = create_dataloaders(
        formatted_annotations, VIDEO_PATH, train_idxs, test_idxs, transformations, batch_size=2
    )
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(
        device
    )
    train(model, train_loader, test_loader, device, annotations)

if __name__ == "__main__":
    run_finetuning(
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
