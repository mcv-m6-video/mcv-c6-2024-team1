from argparse import ArgumentParser
from tqdm import tqdm
import colorsys
import random
import cv2
import numpy as np
import time

from week_utils import *
#from task1_2 import optical_flow_farneback, optical_flow_pyflow


def optical_flow_farneback(
    im1,
    im2,
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
):
    start = time.time()
    flow_farneback = cv2.calcOpticalFlowFarneback(
        im1,
        im2,
        flow=None,
        pyr_scale=pyr_scale,
        levels=levels,
        winsize=winsize,
        iterations=iterations,
        poly_n=poly_n,
        poly_sigma=poly_sigma,
        flags=flags,
    )
    end = time.time()
    total_time = end - start
    return flow_farneback, total_time


def cal_IoU(prev_tl, prev_br, new_tl, new_br):
    # Calculate coordinates of the intersection rectangle
    x_left = max(prev_tl[0], new_tl[0])
    y_top = max(prev_tl[1], new_tl[1])
    x_right = min(prev_br[0], new_br[0])
    y_bottom = min(prev_br[1], new_br[1])

    # If the intersection is valid (non-negative area), calculate the intersection area
    intersection_area = max(0, x_right - x_left + 1) * \
        max(0, y_bottom - y_top + 1)

    # Calculate areas of the individual bounding boxes
    prev_box_area = (prev_br[0] - prev_tl[0] + 1) * \
        (prev_br[1] - prev_tl[1] + 1)
    new_box_area = (new_br[0] - new_tl[0] + 1) * (new_br[1] - new_tl[1] + 1)

    # Calculate the union area
    union_area = prev_box_area + new_box_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou


def generate_random_color():
    # Generate random values for hue, saturation, and value
    hue = random.random()
    # Random saturation between 0.5 and 1.0
    saturation = random.uniform(0.5, 1.0)
    value = random.uniform(0.5, 1.0)  # Random value between 0.5 and 1.0

    # Convert HSV to RGB
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)

    # Convert RGB to BGR
    bgr = tuple(int(255 * x) for x in rgb[::-1])

    return bgr


def task1_3(video_path, annotations_path, video_out_path="", tracking_file="", flow_method="farneback", bbx_flow_method="mean", iou_thr=0.5, visualize=True):
    """
    Object tracking by optical flow
    """

    # Open video file with tqdm
    cap = cv2.VideoCapture(video_path)

    # Check if video file opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        raise

    bbxs = load_json(name=annotations_path)

    if video_out_path != "":
        output_video_filename = f'{flow_method}_{bbx_flow_method}_{video_out_path}'
        codec = cv2.VideoWriter_fourcc(*"XVID")  # Codec for AVI format
        fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter(
            output_video_filename, codec, fps, (frame_width, frame_height)
        )

    if cap.get(cv2.CAP_PROP_FRAME_COUNT) != len(bbxs):
        print("Num of frames in video != num of frames in annotations")
        raise

    track_count = 0
    track_colors = []

    im1 = None
    im2 = None
    all_detections = {}

    for i in tqdm(range(len(bbxs)), desc="Tracking bounding boxes"):
        track_ids = {}

        # Read frame from video
        ret, frame = cap.read()

        # Check if frame was successfully read
        if not ret:
            if i + 1 < len(bbxs):
                print("Error reading frame")
            break

        # Assign a different track to each object on the scene for first frame
        if i == 0:
            for k in bbxs[i]["xmin"]:
                track_ids[k] = track_count
                track_count += 1
                track_colors.append(generate_random_color())
        # Do IoU tracking for second frame
        elif i == 1:
            # Iterate over each bbx
            for k in bbxs[i]["xmin"]:
                max_iou = -1
                max_idx = -1

                new_tl = (bbxs[i]["xmin"][k], bbxs[i]["ymin"][k])
                new_br = (bbxs[i]["xmax"][k], bbxs[i]["ymax"][k])

                # Iterate over each prev bbx
                for j in bbxs[i - 1]["xmin"]:
                    # Check if track has already been assigned
                    if bbxs[i - 1]["track"][j] not in track_ids.values():
                        # Check if both bbxs are same class
                        if bbxs[i]["class"][k] == bbxs[i - 1]["class"][j]:
                            prev_tl = (bbxs[i - 1]["xmin"][j], bbxs[i - 1]["ymin"][j])
                            prev_br = (bbxs[i - 1]["xmax"][j], bbxs[i - 1]["ymax"][j])

                            # Calculate IoU
                            iou = cal_IoU(prev_tl, prev_br, new_tl, new_br)

                            if iou > max_iou and iou > iou_thr:
                                max_iou = iou
                                max_idx = j

                # Check if any bbx passed though the iou thershold
                if max_iou == -1:
                    # New track id
                    track_ids[k] = track_count
                    track_count += 1
                    track_colors.append(generate_random_color())
                else:
                    # Put track id of bbx with max iou
                    track_ids[k] = bbxs[i - 1]["track"][max_idx]
            im1 = im2
        else:
            #if flow_method == "farneback":
            #    flow, _ = optical_flow_farneback(im1, im2)
            #elif flow_method == "pyflow":
            #    im1 = np.atleast_3d(im1.astype(float) / 255.0)
            #    im2 = np.atleast_3d(im2.astype(float) / 255.0)
            #    flow, _ = optical_flow_pyflow(im1, im2)
            #else:
            #    print("Unsupported flow method")
            #    raise
            flow, _ = optical_flow_farneback(im1, im2)

            flow_bbxs = bbxs[i - 1]

            # Shift bounding boxes
            for j in bbxs[i - 1]["xmin"]:
                bbx_flow = flow[int(bbxs[i - 1]["ymin"][j]):int(bbxs[i - 1]["ymax"][j]), int(bbxs[i - 1]["xmin"][j]):int(bbxs[i - 1]["xmax"][j]), :]
                if bbx_flow_method == "mean":
                    x_flow, y_flow = np.mean(bbx_flow, axis=(0, 1))
                if bbx_flow_method == "median":
                    x_flow, y_flow = np.median(bbx_flow, axis=(0, 1))

                flow_bbxs["xmin"][j] += x_flow
                flow_bbxs["xmax"][j] += x_flow
                flow_bbxs["ymin"][j] += y_flow
                flow_bbxs["ymax"][j] += y_flow

            # Iterate over each bbx
            for k in bbxs[i]["xmin"]:
                max_iou = -1
                max_idx = -1

                new_tl = (bbxs[i]["xmin"][k], bbxs[i]["ymin"][k])
                new_br = (bbxs[i]["xmax"][k], bbxs[i]["ymax"][k])

                # Iterate over each shifted bbx
                for j in flow_bbxs["xmin"]:
                    # Check if track has already been assigned
                    if flow_bbxs["track"][j] not in track_ids.values():
                        # Check if both bbxs are same class
                        if bbxs[i]["class"][k] == flow_bbxs["class"][j]:
                            shifted_tl = (flow_bbxs["xmin"][j],
                                       flow_bbxs["ymin"][j])
                            shifted_br = (flow_bbxs["xmax"][j],
                                       flow_bbxs["ymax"][j])

                            # Calculate IoU
                            iou = cal_IoU(shifted_tl, shifted_br, new_tl, new_br)

                            if iou > max_iou and iou > iou_thr:
                                max_iou = iou
                                max_idx = j

                # Check if any bbx passed though the iou thershold
                if max_iou == -1:
                    # New track id
                    track_ids[k] = track_count
                    track_count += 1
                    track_colors.append(generate_random_color())
                else:
                    # Put track id of bbx with max iou
                    track_ids[k] = bbxs[i - 1]["track"][max_idx]

            im1 = im2

        im2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if len(set(track_ids.values())) < len(track_ids):
            print("Repeated track ids")
            raise

        bbxs[i]["track"] = track_ids

        img_draw = frame.copy()
        for k in bbxs[i]["track"]:
            tl = (round(bbxs[i]["xmin"][k]), round(bbxs[i]["ymin"][k]))
            br = (round(bbxs[i]["xmax"][k]), round(bbxs[i]["ymax"][k]))
            img_draw = cv2.rectangle(
                img_draw, (tl), (br), track_colors[bbxs[i]["track"][k]], 2
            )
            img_draw = cv2.putText(
                img_draw,
                str(bbxs[i]["track"][k]),
                (tl),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 0),
                2,
            )

            # Draw circles for previous detections
            if bbxs[i]["track"][k] in all_detections:
                for center in all_detections[bbxs[i]["track"][k]]:
                    img_draw = cv2.circle(
                        img_draw, center, 5, track_colors[bbxs[i]
                                                          ["track"][k]], -1
                    )
            else:
                all_detections[bbxs[i]["track"][k]] = []

            all_detections[bbxs[i]["track"][k]].append(
                (
                    int((bbxs[i]["xmin"][k] + bbxs[i]["xmax"][k]) / 2),
                    int((bbxs[i]["ymin"][k] + bbxs[i]["ymax"][k]) / 2),
                )
            )

        if visualize:
            cv2.imshow(
                "Tracking results",
                cv2.resize(
                    img_draw,
                    (int(img_draw.shape[1] * 0.5),
                     int(img_draw.shape[0] * 0.5)),
                ),
            )
            k = cv2.waitKey(1)
            if k == ord("q"):
                visualize = False

        if video_out_path != "":
            out.write(img_draw)

    if video_out_path != "":
        out.release()
        print(f"Video saved at {video_out_path}")

    if tracking_file != "":
        save_json(bbxs,f'{flow_method}_{bbx_flow_method}_{tracking_file}')


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--predictions-file",
        type=str,
        default="bbxs_clean.json",
        help="Name of prediction json file (YOLO style)",
    )
    parser.add_argument(
        "--tracking-file",
        type=str,
        default="bbxs_clean_tracked.json",
        help="Name of output file",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default="../Data/AICity_data/train/S03/c010/vdo.avi",
        help="Path to video for visualization",
    )
    parser.add_argument(
        "--visualization-path",
        type=str,
        default="tracking.avi",
        help="Path to save visualization video",
    )
    parser.add_argument(
        "--visualize-video",
        type=bool,
        default=False,
        help="Bool to visualize video on execution",
    )
    parser.add_argument(
        "--flow-method",
        type=str,
        default="farneback",
        help="Optical flow method",
    )
    parser.add_argument(
        "--bbx-flow-method",
        type=str,
        default="median",
        help="Method to shift the bounding boxes from the optical flow",
    )
    args = parser.parse_args()

    task1_3(args.video_path, args.predictions_file,
            video_out_path=args.visualization_path, tracking_file=args.tracking_file, visualize=args.visualize_video,
            flow_method=args.flow_method, bbx_flow_method=args.bbx_flow_method)
