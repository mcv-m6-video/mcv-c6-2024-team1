# Setup detectron2 logger
from detectron2.utils.logger import setup_logger

setup_logger()

import sys

import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from tqdm import tqdm

from week_utils import save_json
from argparse import ArgumentParser

CAR_LABEL = 2
DEFAULT_MODEL = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
DEFAULT_MODEL_NAME = "faster_rcnn_R_50_FPN_3x"
VIDEO_PATH = "../Data/AICity_data_S03_C010/AICity_data/train/S03/c010/vdo.avi"


def run_inference(display: bool = False, model=None, name_model=None):
    if not model:
        model = DEFAULT_MODEL
        name_model = DEFAULT_MODEL_NAME

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Error opening video file")

    bbxs = {}
    frame_num = 0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
    cfg.MODEL.DEVICE = "cpu"
    predictor = DefaultPredictor(cfg)

    with tqdm(total=frame_count, file=sys.stdout) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("Error reading frame")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            outputs = predictor(frame)

            for i, bbox in enumerate(outputs["instances"].pred_boxes):
                x_min, y_min, x_max, y_max = bbox.int().cpu().numpy()

                if outputs["instances"].pred_classes[i] == CAR_LABEL:
                    if frame_num not in bbxs:
                        bbxs[frame_num] = []

                    bbxs[frame_num].append(
                        {"bbox": [int(x_min), int(y_min), int(x_max), int(y_max)]}
                    )

                    if display:
                        cv2.rectangle(
                            frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 5
                        )
                        cv2.imshow("Frame", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_num += 1
            pbar.update(1)

    save_json(bbxs, f"results/bbxs_detectron_{name_model}.json")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display video with detections",
        default=False,
    )
    args = parser.parse_args()
    run_inference(display=args.display)
