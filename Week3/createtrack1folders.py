import os
from pathlib import Path
import shutil
import cv2

data_root = "./Data/aic19-track1-train"
trackeval_challenge_root = "./TrackEval/data/gt/mot_challenge"
trackeval_challenge_name = "aicity19"
trackeval_split = "train"

assert Path(trackeval_challenge_root).exists()
assert Path(data_root).exists()

all_seqs = []
for gt_path in Path(data_root).glob("*/*/gt"):
    folder_name = gt_path.parents[1].name + gt_path.parents[0].name
    folder_path = Path(trackeval_challenge_root) / f"{trackeval_challenge_name}-{trackeval_split}" / folder_name / "gt"
    folder_path.mkdir(parents=True, exist_ok=True)
    shutil.copy(gt_path / "gt.txt", folder_path)

    video_file = gt_path.parent / "vdo.avi"
    vid = cv2.VideoCapture(str(video_file))
    length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    with open(folder_path.parent / "seqinfo.ini", "w") as f:
        f.writelines(f"[Sequence]\nname={folder_name}\nseqLength={length}")
    
    all_seqs.append(folder_name)

for split in ["train", "test", "all"]:
    seqmap_file = Path(trackeval_challenge_root) / "seqmaps" / f"{trackeval_challenge_name}-{split}.txt"
    with open(seqmap_file, "w") as f:
        f.write("name\n" + "\n".join(all_seqs))