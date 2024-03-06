import json
from week_utils import readXMLtoAnnotation
import xml.etree.ElementTree as ET
from tqdm import tqdm
from pathlib import Path

bboxes_2_1 = "./Week2/results/bbxs_clean_tracked.json"
out_2_1 = "./Week2/results/bbxs_clean_tracked.txt"

gt_annot = "./Data/ai_challenge_s03_c010-full_annotation.xml"
out_gt = "./Week2/results/s03_gt.txt"

kalman_ins = ["./Week2/results/kalman_new2.json"]
out_folder = "./Week2/results/"

def write_bbxs_to_csv(data, out):
    with open(out, "w") as f:
        for i, frame in tqdm(enumerate(data)):
            written_ids = []
            for obj, track in frame["track"].items():
                if track in written_ids:
                    continue
                written_ids.append(track)
                oframe = {key:frame[key][obj] for key in frame}
                f.write(f'{i+1}, {track}, {oframe["xmin"]}, {oframe["ymin"]}, {oframe["xmax"] - oframe["xmin"]}, {oframe["ymax"] - oframe["ymin"]}, {oframe["confidence"]}, -1, -1, -1\n')


def write_kalman_to_csv(data, out):
    with open(out, "w") as f:
        for frame in tqdm(data):
            written_ids = []
            for track in data[frame]["x_min"]:
                if track in written_ids:
                    continue
                written_ids.append(track)
                oframe = {key:data[frame][key][track] for key in data[frame]}
                f.write(f'{int(frame) + 1}, {track}, {oframe["x_min"]}, {oframe["y_min"]}, {oframe["x_max"] - oframe["x_min"]}, {oframe["y_max"] - oframe["y_min"]}, 1, -1, -1, -1\n')


def XML_to_csv(annots, out, remParked=False):
    file = ET.parse(annots)
    root = file.getroot()

    with open(out, "w") as f:
        for child in tqdm(root):
            if child.tag == "track":
                # Get class
                id = int(child.attrib["id"])
                className = child.attrib["label"]
                for obj in child:
                    if className == "car":
                        objParked = obj[0].text
                        # Do not store if it is parked and we want to remove parked objects
                        if objParked == "true" and remParked:
                            continue
                    
                    frame = obj.attrib["frame"]
                    xtl = float(obj.attrib["xtl"])
                    ytl = float(obj.attrib["ytl"])
                    xbr = float(obj.attrib["xbr"])
                    ybr = float(obj.attrib["ybr"])
                    f.write(f"{int(frame) + 1}, {id}, {xtl}, {ytl}, {xbr - xtl}, {ybr - ytl}, 1, -1, -1, -1\n")


#with open(bboxes_2_1) as f:
#    data = json.load(f)
#write_bbxs_to_csv(data, out_2_1)

for bbxs in kalman_ins:
    out = Path(out_folder) / Path(bbxs).stem / "data" / "S03aicity-01.txt"
    out.parent.mkdir(exist_ok=True, parents=True)
    with open(bbxs) as f:
        data = json.load(f)
    write_kalman_to_csv(data, out)

#XML_to_csv(gt_annot, out_gt)