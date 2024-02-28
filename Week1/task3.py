import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import compute_detections_and_metrics, readVideo, readXMLtoAnnotation

MODEL_NAMES = {0: "MOG", 1: "MOG2", 2: "KNN", 3: "LSBP", 4: "REMBG"}
VIDEO_PATH = "../Data/AICity_data/train/S03/c010/vdo.avi"
ANNOTATIONS_PATH = "../Data/ai_challenge_s03_c010-full_annotation.xml"


def state_of_the_art(vid_path, ind):
    print(f"Selected state of the art model: {MODEL_NAMES[ind]}")

    gray_frames, color_frames = readVideo(vid_path)
    gt = readXMLtoAnnotation(ANNOTATIONS_PATH)

    subs = get_background_subtractor(ind)
    estimation = generate_estimation(subs, gray_frames)
    save_result_image(estimation[560], ind)

    precision_list, recall_list = [], []
    for frame_idx in range(estimation.shape[0]):
        precision, recall = compute_detections_and_metrics(
            frame_idx, estimation[frame_idx], color_frames[frame_idx], gt
        )
        precision_list.append(precision)
        recall_list.append(recall)

    AP = sum(precision_list) / len(precision_list)
    AR = sum(recall_list) / len(recall_list)
    print(f"[Model name: {MODEL_NAMES[ind]}] Average Precision = {AP}")
    print(f"[Model name: {MODEL_NAMES[ind]}] Average Recall = {AR}")


def get_background_subtractor(ind):
    if ind == 0:
        return cv2.bgsegm.createBackgroundSubtractorMOG()
    elif ind == 1:
        return cv2.createBackgroundSubtractorMOG2()
    elif ind == 2:
        return cv2.createBackgroundSubtractorKNN()
    elif ind == 3:
        return cv2.bgsegm.createBackgroundSubtractorLSBP()


def generate_estimation(subs, frames):
    estimation = []
    for frame in frames:
        single_est = subs.apply(frame)
        if len(single_est.shape) == 3:
            single_est = cv2.cvtColor(single_est, cv2.COLOR_RGB2GRAY)
            _, single_est = cv2.threshold(single_est, 200, 255, cv2.THRESH_BINARY)
        boolean_est = single_est.astype(bool)
        estimation.append(boolean_est)

    return np.array(estimation)


def save_result_image(image, ind):
    plt.imshow(image)
    plt.savefig(f"results/state_of_art_{MODEL_NAMES[ind]}")
    plt.close()


if __name__ == "__main__":
    description = """
    INDEXES OF THE SUBTRACTORS:
     \n 0: MOG
     \n 1: MOG2
     \n 2: KNN
     \n 3: LSBP
     """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-i", "--index", type=int, help=description, default=0)
    args = parser.parse_args()
    s_index = args.index

    if s_index in range(5):
        state_of_the_art(VIDEO_PATH, s_index)
    else:
        print("Non-valid Option")
