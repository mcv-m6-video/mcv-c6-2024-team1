from models import *
from utils import *
import json

STORE_VIDEO = True
VIDEO_PATH = "../Data/AICity_data/train/S03/c010/vdo.avi"
ANNOTATIONS_PATH = "../Data/ai_challenge_s03_c010-full_annotation.xml"


def task4():
    gaussian = GaussianColorModel(VIDEO_PATH, color_space=cv2.COLOR_BGR2Lab)
    gaussian.compute_mean_std()
    predictions, frames = gaussian.segment(alpha=4)
    json.dump(predictions, open("predictions_Lab.json", "w"))
    if STORE_VIDEO:
        makeVideo(frames, "video_4_Lab.mp4")
    annots = readXMLtoAnnotation(ANNOTATIONS_PATH, remParked=True)
    annots = removeFirstAnnotations(int(gaussian.num_frames * gaussian.train_split) - 1, annots)
    json.dump(annots, open("annots_Lab.json", "w")) 


if __name__ == "__main__":
    task4()
