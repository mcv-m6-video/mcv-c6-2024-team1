import cv2
import torch

from week_utils import *

TEST_FOLDER = 'test'

REPO_DIR = "ultralytics/yolov5"
MODEL_NAME = "yolov5s"
VIDEO_PATH = "../Data/AICity_data/train/S03/c010/vdo.avi"


def cal_IoU(prev_tl, prev_br, new_tl, new_br):
    # Calculate coordinates of the intersection rectangle
    x_left = max(prev_tl[0], new_tl[0])
    y_top = max(prev_tl[1], new_tl[1])
    x_right = min(prev_br[0], new_br[0])
    y_bottom = min(prev_br[1], new_br[1])

    # If the intersection is valid (non-negative area), calculate the intersection area
    intersection_area = max(0, x_right - x_left + 1) * \
        max(0, y_bottom - y_top + 1)

    # Calculate areas of the individual bounding boxes
    prev_box_area = (prev_br[0] - prev_tl[0] + 1) * \
        (prev_br[1] - prev_tl[1] + 1)
    new_box_area = (new_br[0] - new_tl[0] + 1) * (new_br[1] - new_tl[1] + 1)

    # Calculate the union area
    union_area = prev_box_area + new_box_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou


def main(iou_thr = 0.5, display=False):
    """
    Object tracking by overlap
    """
    # Load YOLOv5s model
    model = torch.hub.load(REPO_DIR, MODEL_NAME, pretrained=True)
    device = torch.device('cuda') if torch.cuda.is_available(
    ) else torch.device('cpu')
    model.to(device)
    model.eval()

    # Open video file
    cap = cv2.VideoCapture(VIDEO_PATH)

    # Check if video file opened successfully
    if not cap.isOpened():
        print("Error opening video file")

    prev_bbxs = {}

    # Loop through each frame of the video
    while cap.isOpened():
        # Create empty list for bounding boxes
        bbxs = []

        # Read frame from video
        ret, frame = cap.read()

        # Check if frame was successfully read
        if not ret:
            print("Error reading frame")
            break

        # Convert frame from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Inference with YOLOv5
        results = model([frame])
        results.render()

        # Add bounding boxes to list
        # bbxs.append(results.pandas().xyxy[0].to_dict())

        if display:
            # Display the frame
            cv2.imshow("Frame", results.ims[0])

            # Check if user pressed the 'q' key to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Obtain car bbxs
        bbxs_car = clean_bbxs(
            [results.pandas().xyxy[0].to_dict()], detected_class="car", confidence=0.5)[0]

        if not bool(prev_bbxs):
            prev_bbxs = bbxs_car
        else:
            max_iou = 0
            max_idx = 0

            for k in prev_bbxs['xmin']:
                prev_tl = (prev_bbxs['xmin'][k], prev_bbxs['ymin'][k])
                prev_br = (prev_bbxs['xmax'][k], prev_bbxs['ymax'][k])
                for j in bbxs_car['xmin']:
                    new_tl = (bbxs_car['xmin'][j], bbxs_car['ymin'][k])
                    new_br = (bbxs_car['xmax'][j], bbxs_car['ymax'][k])

                    # Calculate IoU
                    iou = cal_IoU(prev_tl, prev_br, new_tl, new_br)

                    if iou > max_iou and iou > iou_thr:
                        max_iou = iou
                        max_idx = j

                # Check if any bbx passed though the iou thershold
                if max_iou == 0:
                    # Remove bbx with no new bbx
                    for i in prev_bbxs:
                        prev_bbxs[i].pop(k)
                else:
                    # Populate prev_bbxs
                    prev_bbxs['xmin'][k] = bbxs_car['xmin'][max_idx]
                    prev_bbxs['ymin'][k] = bbxs_car['ymin'][max_idx]
                    prev_bbxs['xmax'][k] = bbxs_car['xmax'][max_idx]
                    prev_bbxs['ymax'][k] = bbxs_car['ymax'][max_idx]

                    # Remove selected bbx
                    for i in bbxs_car:
                        bbxs_car[i].pop(max_idx)


if __name__ == "__main__":
    main()
