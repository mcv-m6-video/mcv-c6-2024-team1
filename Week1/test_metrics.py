from metrics import evaluate, mAP
import json
from argparse import ArgumentParser
STORE_VIDEO = True
VIDEO_PATH = "../Data/AICity_data/train/S03/c010/vdo.avi"
ANNOTATIONS_PATH = "../Data/ai_challenge_s03_c010-full_annotation.xml"


def test(suffix: str):
    predictions = json.load(open(f"predictions/predictions_{suffix}.json", "r"))
    annots = json.load(open("annotations/annots.json", "r"))

    mIoU, precision, recall, f1_score = evaluate(predictions, annots)
    mAP_val = mAP(annots, predictions)
    print(f"mIoU: {mIoU},  mAP: {mAP_val}, precision: {precision}, recall: {recall}, f1_score: {f1_score}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--suffix", type=str, default="rgb")
    args = parser.parse_args()
    test(args.suffix)
