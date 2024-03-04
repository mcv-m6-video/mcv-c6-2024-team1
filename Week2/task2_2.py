import os
import pickle
from sort import Sort

import numpy as np
import cv2
import motmetrics as mm

#from read_data import VideoData, parse_annotations, read_frame_boxes

from week_utils import *

RESULTS_PATH = './results/'
FILE_IN = 'bbxs_clean.json'
FILE_OUT = 'bbxs_clean_tracked.json'

AICITY_DATA_PATH = 'AICity_data/train/S03/c010'

# Read gt file
ANOTATIONS_PATH = 'ai_challenge_s03_c010-full_annotation.xml'
VIDEO_PATH = os.path.join(AICITY_DATA_PATH, "vdo.avi")
GENERATE_VIDEO = True


def load_frame_boxes(path):
    # Get the bboxes
    frame_bboxes = []
    with (open(path, "rb")) as openfile:
        while True:
            try:
                frame_bboxes.append(pickle.load(openfile))
            except EOFError:
                break
    frame_bboxes = frame_bboxes[0]
    return frame_bboxes


def centroid(box):  # box [x,y,w,h]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    x_center = (box[0] + x2) / 2
    y_center = (box[1] + y2) / 2
    return x_center, y_center

def read_boxes_from_file(file_path):

    with open(file_path, 'r') as file:
        frames_boxes = json.load(file)
    return frames_boxes
def convert_to_array(frame):
    x_min = frame['xmin']
    x_max = frame['xmax']
    y_min = frame['ymin']
    y_max = frame['ymax']

    boxes = []
    for id in x_min:
        box = [x_min[id], y_min[id], x_max[id], y_max[id]]
        boxes.append(np.array(box))

    return np.array(boxes)
def obtain_color_pallete(kalman_box_id):
    colors = []
    max_id = max([max(list(k)) for k in kalman_box_id])
    for i in range(int(max_id) + 1):
        color = list(np.random.choice(range(256), size=3))
        colors.append(color)
    return colors

def main():
    print("Load video")
    # Load the boxes from fasterRCNN
    kalman_output = []
    kalman_bboxes = []
    kalman_ids = []
    frame_boxes = read_boxes_from_file(RESULTS_PATH + FILE_IN)

    video_object = cv2.VideoCapture('../Data/AICity_data_S03_C010/AICity_data/train/S03/c010/vdo.avi')

    kalman_tracker = Sort()

    for frame in frame_boxes:
        detected_bboxes = convert_to_array(frame)
        actual_tracks = kalman_tracker.update(detected_bboxes)
        kalman_output.append(actual_tracks)

        #kalman_tracker returns the predicted bounding boxes and the id of the object
        kalman_predicted_bb = actual_tracks[:, 0:4]
        kalman_bboxes.append(kalman_predicted_bb)
        kalman_predicted_id = actual_tracks[:,4]
        kalman_ids.append(kalman_predicted_id)


    if GENERATE_VIDEO:
        frame_width = int(video_object.get(3))
        frame_height = int(video_object.get(4))
        out = cv2.VideoWriter(
            "output_video.mp4",
            cv2.VideoWriter_fourcc(*"MP4V"),
            30,
            (frame_width, frame_height),
        )
        colors = obtain_color_pallete(kalman_ids)
        for f in range(len(kalman_bboxes)):
            ret, frame = video_object.read()
            if not ret:
                break
            for id in zip(kalman_bboxes[f],kalman_ids[f]),:
                print(id)
                break










if __name__ == "__main__":
    main()
