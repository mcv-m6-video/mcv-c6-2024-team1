from pathlib import Path
import subprocess as sp

bbxs_root = "./Week3/bbxs"
sequences_root = "./Data/aic19-track1-train"

for bbx_path in Path(bbxs_root).glob("*.json"):
    sequence = bbx_path.stem.split("_")[-2]
    camera = bbx_path.stem.split("_")[-1]
    sp.run(["C:/Users/goiog/.conda/envs/newC1proj/python.exe",
            "c:/Users/goiog/Desktop/Uni/Master_Computer_Vision/C6/mcv-c6-2024-team1/Week3/task1_3.py",
            "--predictions-file", bbx_path,
            "--tracking-file", f"bbxs_clean_{sequence}_{camera}_tracked.json",
            "--video-path", Path(sequences_root) / sequence / camera / "vdo.avi",
            "--visualization-path", f"tracking_{sequence}_{camera}.avi",
            "--visualize-video", "True"])