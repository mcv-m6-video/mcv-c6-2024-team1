import json

from metrics import evaluate, mAP
from week_utils import readXMLtoAnnotation

ANNOTATIONS_PATH = (
    "../Data/AICity_data_S03_C010/ai_challenge_s03_c010-full_annotation.xml"
)


def test():
    predictions = json.load(open(f"results/bbxs_clean_formatted.json", "r"))
    annots = readXMLtoAnnotation(ANNOTATIONS_PATH, remParked=False)

    mIoU, precision, recall, f1_score = evaluate(predictions, annots)
    mAP_val = mAP(annots, predictions)
    return mIoU, mAP_val, precision, recall, f1_score


if __name__ == "__main__":
    mIou, mAP_val, precision, recall, f1_score = test()
    print(
        f"mIoU: {mIou}, mAP: {mAP_val}, Precision: {precision}, Recall: {recall}, F1 score: {f1_score}"
    )
