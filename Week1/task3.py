import numpy as np
import cv2
from cv2.bgsegm import createBackgroundSubtractorLSBP, createBackgroundSubtractorMOG
import matplotlib.pyplot as plt
import rembg
from PIL import Image
from utils import read_video, read_annotations, split_frames, detect_obj
import pickle
import argparse
import os

# Constants
N_TRAIN_FRAMES = 535
MODEL_NAMES = {0: "MOG", 1: "MOG2", 2: "KNN", 3: "LSBP", 4: "REMBG"}

def state_of_the_art(vid_path, ind):
    print(f"Selected state of the art model: {MODEL_NAMES[ind]}")

    gray_frames, color_frames = read_video(vid_path)
    gt = read_annotations(annotations_path)
    gray_frames_25, gray_frames_75 = split_frames(gray_frames)
    color_frames_25, color_frames_75 = split_frames(color_frames)

    subs = get_background_subtractor(ind)
    estimation, vid = estimate_sota_foreground(subs, gray_frames, ind)
    save_result_image(estimation[560], ind)

    precision_list, recall_list = [], []
    for frame_idx in range(estimation.shape[0]):
        precision, recall = detect_obj(frame_idx, estimation[frame_idx], color_frames[frame_idx], gt)
        precision_list.append(precision)
        recall_list.append(recall)

    AP, AR = calculate_metrics(precision_list, recall_list, ind)
    save_metrics(precision_list, recall_list, ind)

def get_background_subtractor(ind):
    if ind == 0:   return createBackgroundSubtractorMOG()
    elif ind == 1: return cv2.createBackgroundSubtractorMOG2()
    elif ind == 2: return cv2.createBackgroundSubtractorKNN()
    elif ind == 3: return createBackgroundSubtractorLSBP()
    elif ind == 4: return rembg

def estimate_sota_foreground(subs, frames, model_index):
    estimation = []
    estimation_img = []
    for frame in frames:
        single_est = subs.apply(frame)
        if len(single_est.shape) == 3:
            single_est = cv2.cvtColor(single_est, cv2.COLOR_RGB2GRAY)
            _, single_est = cv2.threshold(single_est, 200, 255, cv2.THRESH_BINARY)
        boolean_est = single_est.astype(bool)
        estimation.append(boolean_est)
        estimation_img.append(single_est)
    return np.array(estimation), estimation_img

def save_result_image(image, ind):
    plt.imshow(image)
    plt.savefig(f'results/state_of_art_{MODEL_NAMES[ind]}')

def calculate_metrics(precision_list, recall_list, ind):
    AP = sum(precision_list) / len(precision_list)
    AR = sum(recall_list) / len(recall_list)
    print(f"[Model name: {MODEL_NAMES[ind]}] Average Precision = {AP}")
    print(f"[Model name: {MODEL_NAMES[ind]}] Average Recall = {AR}")
    return AP, AR

def save_metrics(precision_list, recall_list, ind):
    log_AP_name = f'AP_model_{MODEL_NAMES[ind]}.pkl'
    with open(log_AP_name, 'wb') as file:
        pickle.dump({'precision_list': precision_list}, file)
    log_AR_name = f'AR_model_{MODEL_NAMES[ind]}.pkl'
    with open(log_AR_name, 'wb') as file:
        pickle.dump({'recall_list': recall_list}, file)

def use_rembg_outs_for_estimation(path):
    estimations = []
    for i in range(2141):
        img_path = path + "/frame_" + str(i) + ".jpg"
        print(f"Analyzing frame: {i}")
        im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        height, width = im.shape
        single_estimation = [[im[y][x] != 0 for x in range(width)] for y in range(height)]
        estimations.append(single_estimation)
    return np.array(estimations)

def create_rembg_outs(frame, i):
    input_array = np.array(frame)
    output_array = rembg.remove(input_array, session=rembg_session, alpha_matting=True,
                                 alpha_matting_foreground_threshold=270, alpha_matting_background_threshold=20,
                                 alpha_matting_erode_size=11)
    output_image = Image.fromarray(output_array)
    rgb_im = output_image.convert('RGB')
    save_path = rembg_output_path + "frame_" + str(i) + ".jpg"
    rgb_im.save(save_path)

if __name__ == "__main__":
    description = """
    INDEXES OF THE SUBTRACTORS:
     \n 0: MOG
     \n 1: MOG2
     \n 2: KNN
     \n 3: LSBP
     \n 4: rembg (special)
     """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--index', type=int, help=description, default=0)
    args = parser.parse_args()
    s_index = args.index

    vid_path = "../Data/AICity_data/train/S03/c010/vdo.avi"
    annotations_path = "../Data/ai_challenge_s03_c010-full_annotation.xml"
    rembg_output_path = "/output_task3/"

    if s_index in range(5):
        state_of_the_art(vid_path, s_index)
    else:
        print('Non-valid Option')
