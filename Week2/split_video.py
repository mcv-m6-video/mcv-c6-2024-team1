import os
import cv2
from tqdm import tqdm

VIDEO_PATH = "../Data/AICity_data_S03_C010/AICity_data/train/S03/c010/vdo.avi"
os.makedirs('video_frames', exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
for idx in tqdm(range(2141)):
    _, frame = cap.read()
    cv2.imwrite(os.path.join('video_frames',f'f_{idx}.jpg'), frame)