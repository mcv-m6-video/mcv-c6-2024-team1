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


def save_json(file: dict, name: str):
    with open(name, "w") as f:
        json.dump(file, f)
