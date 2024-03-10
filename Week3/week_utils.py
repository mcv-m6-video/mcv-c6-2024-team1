import json
import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from tqdm import tqdm

DEFAULT_VIDEO_PATH = "../Data/AICity_data_S03_C010/AICity_data/train/S03/c010/vdo.avi"


def split_video(video_path: str = DEFAULT_VIDEO_PATH):
    os.makedirs("video_frames", exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    for idx in tqdm(range(2141)):
        _, frame = cap.read()
        cv2.imwrite(os.path.join("video_frames", f"f_{idx}.jpg"), frame)


def clean_bbxs(bbxs: list, detected_class: str, confidence: float = 0.5):
    bbxs_clean = []

    for entry in bbxs:
        new_entry = {key: {} for key in entry}
        for i in entry["name"]:
            if (
                entry["name"][i] == detected_class
                and entry["confidence"][i] >= confidence
            ):
                for key in entry:
                    new_entry[key][i] = entry[key][i]
        bbxs_clean.append(new_entry)

    return bbxs_clean


def load_json(name: str):
    with open(name, "r") as f:
        return json.load(f)


def save_json(file: dict, name: str):
    with open(name, "w") as f:
        json.dump(file, f)


def display_video_with_detections(
    video_path: str = DEFAULT_VIDEO_PATH, store_video: bool = True
):
    current_frame = 0
    bbxs = load_json("results/bbxs_detectron_faster_rcnn_R_50_FPN_3x.json")
    cap = cv2.VideoCapture(video_path)
    annotations = load_json("../Week1/annotations/annots.json")
    if store_video:
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter(
            "output_video.mp4",
            cv2.VideoWriter_fourcc(*"MP4V"),
            30,
            (frame_width, frame_height),
        )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_bbxs = bbxs[str(current_frame)]

        if current_frame >= 535:
            if str(current_frame) in annotations:
                current_bbxs_annot = annotations[str(current_frame)]
            else:
                current_bbxs_annot = []

            for i, bbox in enumerate(current_bbxs):
                xmin = bbox["bbox"][0]
                ymin = bbox["bbox"][1]
                xmax = bbox["bbox"][2]
                ymax = bbox["bbox"][3]
                if i < len(current_bbxs_annot):
                    xmin_anns = int(current_bbxs_annot[i]["bbox"][0])
                    ymin_anns = int(current_bbxs_annot[i]["bbox"][1])
                    xmax_anns = int(current_bbxs_annot[i]["bbox"][2])
                    ymax_anns = int(current_bbxs_annot[i]["bbox"][3])
                    cv2.rectangle(
                        frame,
                        (xmin_anns, ymin_anns),
                        (xmax_anns, ymax_anns),
                        (0, 0, 255),
                        2,
                    )
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            cv2.imshow("Frame", frame)
            if store_video:
                out.write(frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        current_frame += 1


def convert_bbxs_to_annots_format(bbxs: list):
    frame_data = {}
    for frame, entry in enumerate(bbxs):
        for i in entry["xmin"]:
            bbox = [
                entry["xmin"][i],
                entry["ymin"][i],
                entry["xmax"][i],
                entry["ymax"][i],
            ]
            if str(frame) not in frame_data:
                frame_data[str(frame)] = []

            frame_data[str(frame)].append({"bbox": bbox})

    return frame_data


def readXMLtoAnnotation(annotationFile, remParked=False):
    # Read XML
    file = ET.parse(annotationFile)
    root = file.getroot()
    annotations = {}

    # Find objects
    for child in root:
        if child.tag == "track":
            # Get class
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
                bbox = [xtl, ytl, xbr, ybr]
                if frame in annotations:
                    annotations[frame].append({"name": className, "bbox": bbox})
                else:
                    annotations[frame] = [{"name": className, "bbox": bbox}]

    return annotations


def convertAnnotations(annotations):
    new_annotations = {}
    for frame in annotations:
        bbxs = []
        for obj in annotations[frame]:
            obj["bbox"] = [int(x) for x in obj["bbox"]]
            bbxs.append(obj["bbox"])
        new_annotations[int(frame)] = bbxs
    return new_annotations


def split_strategy_A(number_of_frames):
    FRAME_25_PERCENT = int(number_of_frames / 4)
    train_idxs = np.arange(0, FRAME_25_PERCENT)
    test_idxs = np.arange(FRAME_25_PERCENT, number_of_frames)
    return train_idxs, test_idxs


def split_strategy_B(fold_idx, number_of_frames, k=4):
    # Assuming 4 folds
    fold_size = number_of_frames / k
    train_idxs = []
    test_idxs = []
    for i in range(k):
        if i == fold_idx:
            train_idxs = np.arange(i * fold_size, (i + 1) * fold_size)
        else:
            test_idxs.extend(np.arange(i * fold_size, (i + 1) * fold_size))
    return train_idxs, test_idxs


def split_strategy_C(number_of_frames):
    train_size = int(0.25 * number_of_frames)
    train_idxs = np.random.choice(number_of_frames, train_size, replace=False)
    test_idxs = np.setdiff1d(np.arange(number_of_frames), train_idxs)
    return train_idxs, test_idxs


def compute_flow_metrics(flow, gt):
    # Binary mask to discard non-occluded areas
    # non_occluded_areas = gt[:,:,2] != 0

    # Only for the first 2 channels
    square_error_matrix = (flow[:, :, 0:2] - gt[:, :, 0:2]) ** 2
    square_error_matrix_valid = square_error_matrix * np.stack(
        (gt[:, :, 2], gt[:, :, 2]), axis=2
    )
    # square_error_matrix_valid = square_error_matrix[non_occluded_areas]

    # non_occluded_pixels = np.shape(square_error_matrix_valid)[0]
    non_occluded_pixels = np.sum(gt[:, :, 2] != 0)

    # Compute MSEN
    pixel_error_matrix = np.sqrt(
        np.sum(square_error_matrix_valid, axis=2)
    )  # Pixel error for both u and v
    msen = (1 / non_occluded_pixels) * np.sum(
        pixel_error_matrix
    )  # Average error for all non-occluded pixels

    # Compute PEPN
    erroneous_pixels = np.sum(pixel_error_matrix > 3)
    pepn = erroneous_pixels / non_occluded_pixels

    return msen, pepn, pixel_error_matrix


def load_flow_gt(flow_path):
    flow_raw = cv2.imread(flow_path, cv2.IMREAD_UNCHANGED).astype(np.double)

    # Transform data (DevKit Stereo Flow - KITTI)
    flow_u = (flow_raw[:, :, 2] - 2**15) / 64.0
    flow_v = (flow_raw[:, :, 1] - 2**15) / 64.0
    flow_valid = flow_raw[:, :, 0] == 1

    # Set to 0 the points where the flow is not valid
    flow_u[~flow_valid] = 0
    flow_v[~flow_valid] = 0

    # Reorder channels
    return np.stack((flow_u, flow_v, flow_valid), axis=2)


def hsv_plot(flow):
    w, h, _ = flow.shape
    hsv = np.zeros((w, h, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def save_video(images, video_name, fps=30):
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        video_name, fourcc, fps, (images[0].shape[1], images[0].shape[0]), True
    )

    for i in range(len(images)):
        out.write(images[i].astype(np.uint8))

    out.release()
