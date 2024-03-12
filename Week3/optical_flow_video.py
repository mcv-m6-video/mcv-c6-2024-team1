from argparse import ArgumentParser

import cv2
import numpy as np
from tqdm import tqdm

from task1_2 import optical_flow_farneback, optical_flow_pyflow
from week_utils import hsv_plot, save_video


def make_video(input_video_path, output_video_path, flow_method="pyflow"):
    frames = []
    curr_frame = 0
    cap = cv2.VideoCapture(input_video_path)
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    pbar = tqdm(total=num_frames, desc="Processing frames")

    while curr_frame < num_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        ret, im1 = cap.read()
        if not ret:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame + 1)
        ret, im2 = cap.read()
        if not ret:
            break

        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        if flow_method == "pyflow":
            im1 = np.atleast_3d(im1.astype(float) / 255.0)
            im2 = np.atleast_3d(im2.astype(float) / 255.0)
            flow, _ = optical_flow_pyflow(im1, im2)
        elif flow_method == "farneback":
            flow, _ = optical_flow_farneback(im1, im2)

        curr_frame += 1
        pbar.update(1)
        frames.append(np.array(hsv_plot(flow)))

    save_video(frames, output_video_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--flow-method",
        type=str,
        default="farneback",
        help="Optical flow method used for the video",
        choices=["pyflow", "farneback"],
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="../Data/AICity_data_S03_C010/AICity_data/train/S03/c010/vdo.avi",
    )
    parser.add_argument("--output-path", type=str, default="output_video.mp4")

    args = parser.parse_args()
    make_video(
        input_video_path=args.input_path,
        output_video_path=args.output_path,
        flow_method=args.flow_method,
    )
