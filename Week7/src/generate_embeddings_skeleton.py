import torch
import pickle
from tqdm import tqdm
from mmaction.apis import inference_recognizer, init_recognizer

ANNOTS_PATH = "/ghome/group01/mcv-c6-2024-team1/Week7/data/hmdb51_2d.pkl"
EMBEDDINGS_PATH = "/ghome/group01/mcv-c6-2024-team1/Week7/results/embeddings_posec3d_train1"

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

    model = init_recognizer(CONFIG_PATH, CHECKPOINT_PATH, device="cuda:0" if torch.cuda.is_available() else "cpu")
    avg_pool = model.cls_head.avg_pool
    model.cls_head = None
    
    ids = set([annot for annot in annots["split"]["train1"]]) # Define split
    annotations = {annot["frame_dir"]: annot for annot in annots["annotations"] if annot["frame_dir"] in ids}
    keys = list(annotations.keys())

    batch = keys[len(keys) // 2:] # Define batch
    batch_embeddings = {}

    for key in tqdm(batch, desc="Generating embeddings..."):
        embed = inference_recognizer(model, annotations[key])
        embed = avg_pool(embed)
        embed = embed.view(embed.shape[0], -1).squeeze()
        batch_embeddings.update({key: embed.cpu().numpy()})

    save(batch_embeddings, EMBEDDINGS_PATH + "_2.pkl") # Define X
    batch_embeddings.clear()