from pathlib import Path
import os
from pathlib import Path
import shutil
import cv2

CSV_FOLDER = Path("./Week4/EvaluationData")
TRACKER_FOLDER = "./TrackEval/data/trackers/mot_challenge/mtmc-train"
TRACKER_NAME = "similarity"

CHALLENGE_ROOT = "./TrackEval/data/gt/mot_challenge"
CHALLENGE_NAME = "mtmc"
TRACKEVAL_SPLIT = "train"
SEQ_NAME = f"{CHALLENGE_NAME}01"

assert Path(CHALLENGE_ROOT).exists()

with open(CSV_FOLDER / "S03_gt.csv") as f:
    lines_gt = [line.strip().split(",") for line in f.readlines()[1:]]
with open(CSV_FOLDER / "S03_tracklets.csv") as f:
    lines_pred = [line.strip().split(",") for line in f.readlines()[1:]]
    lines_pred = [[str(int(line[0]) + 1), line[5]] + line[1:5] + ["1", "-1", "-1", "-1"] for line in lines_pred]
length = int(max(lines_gt + lines_pred, key=lambda k: int(k[0]))[0])
print(length)
folder_path = Path(CHALLENGE_ROOT) / f"{CHALLENGE_NAME}-{TRACKEVAL_SPLIT}" / SEQ_NAME / "gt"
folder_path.mkdir(parents=True, exist_ok=True)
with open(folder_path / "gt.txt", "w") as f:
    f.writelines([" ".join(line) + "\n" for line in lines_gt])

with open(folder_path.parent / "seqinfo.ini", "w") as f:
        f.write(f"[Sequence]\nname={SEQ_NAME}\nseqLength={length}")

for split in ["train", "test", "all"]:
    seqmap_file = Path(CHALLENGE_ROOT) / "seqmaps" / f"{CHALLENGE_NAME}-{split}.txt"
    with open(seqmap_file, "w") as f:
        f.write(f"name\n{SEQ_NAME}")

seq_file = Path(TRACKER_FOLDER) / TRACKER_NAME / "data" / f"{SEQ_NAME}.txt"
seq_file.parent.mkdir(parents=True, exist_ok=True)

with open(seq_file, "w") as f:
    f.writelines([" ".join(line) + "\n" for line in lines_pred])
