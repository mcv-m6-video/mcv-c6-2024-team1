import json

from metrics import evaluate, mAP
from argparse import ArgumentParser

STORE_VIDEO = True
VIDEO_PATH = "../Data/AICity_data/train/S03/c010/vdo.avi"
ANNOTATIONS_PATH = "../Data/ai_challenge_s03_c010-full_annotation.xml"
SUFFIXES = ["default", "adaptative", "hsv", "lab", "rgb", "YCrCb", "yuv", "2_hsv", "2_lab", "2_rgb", "2_YCrCb"]

def test(suffix: str):
    predictions = json.load(open(f"predictions/predictions_{suffix}.json", "r"))
    annots = json.load(open("annotations/annots.json", "r"))

    mIoU, precision, recall, f1_score = evaluate(predictions, annots)
    mAP_val = mAP(annots, predictions)
    return mIoU, mAP_val, precision, recall, f1_score

def test_all():
    with open("results.json", "w") as f:
        results = {}
        for suffix in SUFFIXES:
            mIoU, mAP_val, precision, recall, f1_score = test(suffix)
            results[suffix] = {
                "mIoU": mIoU,
                "mAP": mAP_val,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score
            }
        json.dump(results, f)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--suffix", type=str, default="rgb")
    args = parser.parse_args()
    
    if args.suffix == "all":
        test_all()
    else:
        test(args.suffix)
