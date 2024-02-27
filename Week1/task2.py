import json

from models import *
from utils import *

STORE_VIDEO = False
VIDEO_PATH = "../Data/AICity_data/train/S03/c010/vdo.avi"
ANNOTATIONS_PATH = "../Data/ai_challenge_s03_c010-full_annotation.xml"


def task2():
    gaussian = AdaptativeGaussianModel(VIDEO_PATH)
    gaussian.compute_mean_std()
    predictions, frame, bg, det = gaussian.segment(alpha=4)
    json.dump(predictions, open("predictions/predictions_adaptative.json", "w"))
    if STORE_VIDEO:
        makeVideo(frame, "video.mp4")
        makeVideo(bg, "video_bg.mp4")
        makeVideo(det, "video_det.mp4")
    annots = readXMLtoAnnotation(ANNOTATIONS_PATH, remParked=True)
    annots = removeFirstAnnotations(
        int(gaussian.num_frames * gaussian.train_split) - 1, annots
    )


if __name__ == "__main__":
    task2()
