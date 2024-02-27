from models import *
from utils import *
import json

STORE_VIDEO = True
VIDEO_PATH = "./Data/AICity_data/train/S03/c010/vdo.avi"
ANNOTATIONS_PATH = "./Data/ai_challenge_s03_c010-full_annotation.xml"


def task4_2():
    gaussian = AdaptativeGaussianColorModel(
        VIDEO_PATH, color_space=cv2.COLOR_BGR2HSV, reverse_color_space=cv2.COLOR_HSV2BGR)
    gaussian.compute_mean_std()
    predictions, frames, bg, det = gaussian.segment(alpha=4)
    json.dump(predictions, open("./Week1/predictions/predictions_HSV.json", "w"))
    if STORE_VIDEO:
        makeVideo(frames, "video_4_2_HSV.mp4")
        makeVideo(bg, "video_bg_4_2_HSV.mp4")
        makeVideo(det, "video_det_4_2_HSV.mp4")
    annots = readXMLtoAnnotation(ANNOTATIONS_PATH, remParked=True)
    annots = removeFirstAnnotations(
        int(gaussian.num_frames * gaussian.train_split) - 1, annots
    )
    json.dump(annots, open("./Week1/annotations/annots_HSV.json", "w"))


if __name__ == "__main__":
    task4_2()
