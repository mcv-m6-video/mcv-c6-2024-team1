from detectron2.utils.logger import setup_logger

setup_logger()

import os, cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from tqdm import tqdm

MODEL = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
INPUT_DIR = "/ghome/group01/MCV-C6/video_frames"
OUTPUT_DIR = "./m6-experiments"
DATASET_NAME = "m6-aicity"
VIDEO_PATH = 'vdo.avi'

def run_inference():
    experiment_name = f"{DATASET_NAME}_{MODEL[15:-5]}"
    output_dir = os.path.join(OUTPUT_DIR, experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MODEL))
    cfg.OUTPUT_DIR = output_dir
    cfg.DATASETS.TRAIN = (DATASET_NAME,)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL)
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

    print("Evaluating...")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    with open(experiment_name + ".txt", "w") as fp:
        cap = cv2.VideoCapture(VIDEO_PATH)
        num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        for _ in tqdm(range(int(num_frames))):
            frame_num, im = cap.read()
            print(f"Processing frame {frame_num}")
            if im is None:
                print("Error reading image")
                continue
            out = predictor(im)
            inst = out["instances"]
            inst = inst[inst.pred_classes == 2]
            print(inst)
            for bbox, conf in zip(inst.pred_boxes, inst.scores):
                bbox = bbox.to("cpu").numpy()
                conf = conf.to("cpu").numpy()
                x1, y1, x2, y2 = bbox
                #   'frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z'
                line = f"{frame_num}, -1, {x1}, {y1}, {x2-x1}, {y2-y1}, {conf}, -1, -1, -1\n"
                fp.write(line)
    print("Done")


if __name__ == "__main__":
    run_inference()
