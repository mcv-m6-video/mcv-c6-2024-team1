import pickle
import torch
from tqdm import tqdm
from mmaction.apis import inference_recognizer, init_recognizer

ANNOTS_PATH = "/ghome/group01/mcv-c6-2024-team1/Week7/data/hmdb51_2d.pkl"
RESULTS_PATH = "/ghome/group01/mcv-c6-2024-team1/Week7/results/results_x3d.pkl"
MMACTIONS_PATH = "/ghome/group01/mcv-c6-2024-team1/Week7/results/results_posec3d.pkl"

CONFIG_PATH = "/ghome/group01/mcv-c6-2024-team1/Week7/mmaction2/configs/skeleton/posec3d/slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_hmdb51-split1-keypoint.py"
CHECKPOINT_PATH = "/ghome/group01/mcv-c6-2024-team1/Week7/mmaction2/work_dirs/slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_hmdb51-split1-keypoint/best_acc_top1_epoch_10.pth"

def load(path: str) -> object:
    with open(path, "rb") as f:
        pkl = pickle.load(f)

    return pkl

def save(obj: dict, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

if __name__ == "__main__":
    annots = load(ANNOTS_PATH)
    results = load(RESULTS_PATH)

    model = init_recognizer(CONFIG_PATH, CHECKPOINT_PATH, device="cuda:0" if torch.cuda.is_available() else "cpu")

    ids = set([annot for annot in annots["split"]["test1"]])
    annotations = {annot["frame_dir"]: annot for annot in annots["annotations"] if annot["frame_dir"] in ids}
    mmaction_results = {}

    for key in tqdm(results.keys(), desc="Running inference..."):
        result = inference_recognizer(model, annotations[key])
        mmaction_results.update({key: result.pred_score.cpu().numpy()})

    save(mmaction_results, MMACTIONS_PATH)