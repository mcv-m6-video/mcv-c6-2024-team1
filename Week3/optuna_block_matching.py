import cv2
import numpy as np
from PIL import Image
import optuna
from time import time

from week_utils import compute_flow_metrics, load_flow_gt
from bm_utils import ssd, sad, ncc
from block_matching import block_match_log

sequence_number = "45"
CURRENT_IMAGE = f"../Data/data_stereo_flow/training/image_0/0000{sequence_number}_10.png"
REF_IMAGE = f"../Data/data_stereo_flow/training/image_0/0000{sequence_number}_11.png"
GT_PATH = f"../Data/data_stereo_flow/training/flow_noc/0000{sequence_number}_10.png"

curr = np.asarray(Image.open(CURRENT_IMAGE))
ref = np.asarray(Image.open(REF_IMAGE))

distances_dict = {
    "ssd": ssd,
    "sad": sad,
    "ncc": ncc
}
distance_options = list(distances_dict.keys())

def objective(trial: optuna.Trial):

    block_size = trial.suggest_int("block_size", 1, 180)
    # search area must be greater than block size
    search_area = trial.suggest_int("search_area", block_size+1, 200)
    distance = trial.suggest_categorical("distance", [
        "ssd",
        "sad",
        "ncc"
    ])
    selected_dist = distances_dict[distance]
    direction = trial.suggest_categorical("direction", [
        "forward",
        "backward"
    ])
    max_level = trial.suggest_int("max_level", 1, 5)

    initial_time = time()
    flow = block_match_log(ref, curr, (block_size, block_size), (search_area, search_area), selected_dist, direction, max_level)
    final_time = time()

    gt_flow, flow_u, _ = load_flow_gt(GT_PATH)
    flow_reshaped = cv2.resize(flow, (flow_u.shape[1], flow_u.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
    # Calculate MSEN and PEPN
    msen, pepn, _ = compute_flow_metrics(flow_reshaped, gt_flow)

    return msen

study = optuna.create_study(study_name="block_matching_msen_x8",
                            storage="sqlite:///block_matching.db",
                            load_if_exists=True)
study.optimize(objective, n_trials=500)


