from models import *
from utils import *
from metrics import evaluate
import optuna
import json

VIDEO_PATH = "../Data/AICity_data/train/S03/c010/vdo.avi"

ANNOTATIONS_PATH = "annotations/annots.json"
ANNOTATIONS = json.load(open(ANNOTATIONS_PATH, "r"))

# Best parameters found using Optuna
BEST_PARAMS = {
    "kernel_open_size": 3,
    "kernel_close_size": 30,
    "area_threshold": 5000,
    "alpha": 7,
}


def objective(trial):
    gaussian = AdaptativeGaussianModel(
        video_path=VIDEO_PATH,
        kernel_open_size=BEST_PARAMS["kernel_open_size"],
        kernel_close_size=BEST_PARAMS["kernel_close_size"],
        area_threshold=BEST_PARAMS["area_threshold"],
        rho=trial.suggest_categorical("rho", [0.05, 0.1, 0.2, 0.3, 0.5, 0.6]),
        median_filter_before=trial.suggest_categorical(
            "median_filter_before", [None, 3, 7, 15]
        ),
        median_filter_after=trial.suggest_categorical(
            "median_filter_after", [None, 3, 7, 15]
        ),
        use_mask=trial.suggest_categorical("use_mask", [False, True]),
    )
    gaussian.compute_mean_std()
    predictions, _, _, _ = gaussian.segment(alpha=BEST_PARAMS["alpha"])

    mIoU, precision, recall, f1_score = evaluate(predictions, ANNOTATIONS)
    return mIoU + precision + recall + f1_score


search_space = {
    "median_filter_before": [None, 3, 7, 15],
    "median_filter_after": [None, 3, 7, 15],
    "rho": [0.05, 0.1, 0.2, 0.3, 0.5, 0.6],
    "use_mask": [False, True],
}

study = optuna.create_study(
    sampler=optuna.samplers.GridSampler(search_space),
    direction="maximize",  # redundand, since grid search
    storage="sqlite:///iou_segmentation_adaptative.db",
    study_name="1_non_adaptative_part1",
)
study.optimize(objective)
