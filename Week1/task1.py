import json

from models import *
from utils import *

STORE_VIDEO = False
VIDEO_PATH = "../Data/AICity_data/train/S03/c010/vdo.avi"
ANNOTATIONS_PATH = "../Data/ai_challenge_s03_c010-full_annotation.xml"


def task1():
    gaussian = GaussianModel(VIDEO_PATH)
    gaussian.compute_mean_std()
    predictions, frames, det = gaussian.segment(alpha=4)
    json.dump(predictions, open("predictions/predictions_default.json", "w"))
    if STORE_VIDEO:
        makeVideo(frames, "video_t1.mp4")
        makeVideo(det, "video_det_t1.mp4")
    annots = readXMLtoAnnotation(ANNOTATIONS_PATH, remParked=True)
    annots = removeFirstAnnotations(
        int(gaussian.num_frames * gaussian.train_split) - 1, annots
    )
    # json.dump(annots, open("annotations/annots.json", "w"))


if __name__ == "__main__":
    task1()
