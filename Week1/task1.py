from models import *
from utils import *
import json

VIDEO_PATH = "../Data/AICity_data/train/S03/c010/vdo.avi"
ANNOTATIONS_PATH = "../Data/ai_challenge_s03_c010-full_annotation.xml"


def task1():
    gaussian = GaussianModel(VIDEO_PATH)
    gaussian.compute_mean_std()
    predictions = gaussian.segment(alpha = 11)
    # json.dump(predictions, open("predictions_new.json", "w"))

    annots, imageNames = readXMLtoAnnotation(ANNOTATIONS_PATH, remParked = True)
    annots, imageNames = removeFirstAnnotations(552, annots, imageNames)
    # json.dump(annots, open("annots.json", "w"))


    
if __name__ == "__main__":
    task1()