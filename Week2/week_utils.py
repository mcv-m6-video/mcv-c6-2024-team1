import json
import os

import cv2
from tqdm import tqdm

DEFAULT_VIDEO_PATH = "../Data/AICity_data_S03_C010/AICity_data/train/S03/c010/vdo.avi"


def split_video(video_path: str = DEFAULT_VIDEO_PATH):
    os.makedirs("video_frames", exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    for idx in tqdm(range(2141)):
        _, frame = cap.read()
        cv2.imwrite(os.path.join("video_frames", f"f_{idx}.jpg"), frame)


def clean_bbxs(bbxs: list, detected_class: str, confidence: float = 0.5):
    bbxs_clean = []

    for entry in bbxs:
        new_entry = {key: {} for key in entry}
        for i in entry["name"]:
            if (
                entry["name"][i] == detected_class
                and entry["confidence"][i] >= confidence
            ):
                for key in entry:
                    new_entry[key][i] = entry[key][i]
        bbxs_clean.append(new_entry)

    return bbxs_clean


def load_json(name: str):
    with open(name, "r") as f:
        return json.load(f)


def save_json(file: dict, name: str):
    with open(name, "w") as f:
        json.dump(file, f)


def display_video_with_detections(
    video_path: str = DEFAULT_VIDEO_PATH, store_video: bool = False
):
    current_frame = 0
    bbxs = load_json("bbxs_clean.json")
    cap = cv2.VideoCapture(video_path)

    if store_video:
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter(
            "output_video.mp4",
            cv2.VideoWriter_fourcc(*"MP4V"),
            30,
            (frame_width, frame_height),
        )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_bbx = bbxs[current_frame]
        names = current_bbx["name"]
        for key in names:
            xmin = int(current_bbx["xmin"][key])
            ymin = int(current_bbx["ymin"][key])
            xmax = int(current_bbx["xmax"][key])
            ymax = int(current_bbx["ymax"][key])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        cv2.imshow("Frame", frame)
        if store_video:
            out.write(frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        current_frame += 1


display_video_with_detections(store_video=True)