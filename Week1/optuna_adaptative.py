from models import *
from utils import *
from metrics import mIoU
import optuna
import json

VIDEO_PATH = "../Data/AICity_data/train/S03/c010/vdo.avi"

ANNOTATIONS_PATH = "annotations/annots.json"
ANNOTATIONS = json.load(open(ANNOTATIONS_PATH, "r"))


def objective(trial):
    gaussian = GaussianModel(
        video_path=VIDEO_PATH,
        kernel_open_size=trial.suggest_categorical("kernel_open_size", [3, 5, 10]),
        kernel_close_size=trial.suggest_categorical(
            "kernel_close_size", [3, 5, 10, 20, 30]
        ),
        area_threshold=trial.suggest_categorical(
            "area_threshold", [1000, 2000, 3000, 4000, 5000, 6000]
        ),
    )
    gaussian.compute_mean_std()
    predictions, _ = gaussian.segment(alpha=trial.suggest_float("alpha", 2, 11))

    return mIoU(predictions, ANNOTATIONS)


search_space = {
    "kernel_open_size": [3, 5, 10],
    "kernel_close_size": [3, 5, 10, 20, 30],
    "area_threshold": [1000, 2000, 3000, 4000, 5000, 6000],
    "alpha": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
}

study = optuna.create_study(
    sampler=optuna.samplers.GridSampler(search_space),
    direction="maximize",  # redundand, since grid search
    storage="sqlite:///iou_segmentation.db",
    study_name="1",
)
study.optimize(objective)
