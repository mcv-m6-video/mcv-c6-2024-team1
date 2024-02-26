from models import *
from utils import *
import json

STORE_VIDEO = True
VIDEO_PATH = "../Data/AICity_data/train/S03/c010/vdo.avi"
ANNOTATIONS_PATH = "../Data/ai_challenge_s03_c010-full_annotation.xml"


def task4():
    gaussian = GaussianMixtureModel(VIDEO_PATH)
    gaussian.compute_mean_std()
    predictions, frames = gaussian.segment(alpha=4)
    json.dump(predictions, open("predictions.json", "w"))
    if STORE_VIDEO:
        makeVideo(frames, "video4.mp4")
    annots, imageNames = readXMLtoAnnotation(ANNOTATIONS_PATH, remParked=True)
    annots, imageNames = removeFirstAnnotations(
        int(gaussian.num_frames * gaussian.train_split) - 1, annots, imageNames)
    json.dump(annots, open("annots.json", "w")) 


if __name__ == "__main__":
    task4()
