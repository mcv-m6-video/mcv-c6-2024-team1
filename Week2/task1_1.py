import cv2
import torch

from week_utils import *

REPO_DIR = "ultralytics/yolov5"
MODEL_NAME = "yolov5s"
VIDEO_PATH = "../Data/AICity_data_S03_C010/AICity_data/train/S03/c010/vdo.avi"


def run_inference(display: bool = False):
    # Load YOLOv5s model
    model = torch.hub.load(REPO_DIR, MODEL_NAME, pretrained=True)

    # Open video file
    cap = cv2.VideoCapture(VIDEO_PATH)

    # Check if video file opened successfully
    if not cap.isOpened():
        print("Error opening video file")

    # Create empty list for bounding boxes
    bbxs = []

    # Loop through each frame of the video
    while cap.isOpened():
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
        bbxs.append(results.pandas().xyxy[0].to_dict())

        if display:
            # Display the frame
            cv2.imshow("Frame", results.ims[0])

            # Check if user pressed the 'q' key to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    bbxs_clean = clean_bbxs(bbxs, detected_class="car", confidence=0.5)

    # Save bounding boxes and clean bounding boxes to JSON files
    save_json(bbxs, "bbxs.json")
    save_json(bbxs_clean, "bbxs_clean.json")
    converted_bbxs = convert_bbxs_format(bbxs_clean)
    save_json(converted_bbxs, "bbxs_clean_formatted.json")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_inference()
