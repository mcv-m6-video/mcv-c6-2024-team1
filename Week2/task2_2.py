import os
import pickle
from sort import *

import numpy as np
import cv2
import motmetrics as mm
from visualize_tracking import *

#from read_data import VideoData, parse_annotations, read_frame_boxes

from week_utils import *

SAVE = False
VISUALIZE = True

RESULTS_PATH = './results/'
FILE_IN = 'bbxs_clean.json'
FILE_OUT = 'kalman.json'

AICITY_DATA_PATH = 'AICity_data/train/S03/c010'
VIDEO_OUT_PATH = './results/kalman_tracking_vdo.avi'

# Read gt file
ANOTATIONS_PATH = 'ai_challenge_s03_c010-full_annotation.xml'
VIDEO_PATH = '../Data/AICity_data/train/S03/c010/vdo.avi'
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
    score = frame['confidence']

    boxes = []
    for id in x_min:
        box = [x_min[id], y_min[id], x_max[id], y_max[id],score[id] ]
        boxes.append(np.array(box))

    return np.array(boxes)

def obtain_color_pallete(kalman_box_id):
    colors = []
    max_id = max([max(list(k)) for k in kalman_box_id])
    for i in range(int(max_id) + 1):
        color = list(np.random.choice(range(256), size=3))
        colors.append(color)
    return colors

def recover_detection_ids(initialbbox,ids,kalmanbbox):
    tracker={}
    for id,box in zip(ids,initialbbox):
        print(box)
        x_min_i,y_min_i,x_max_i,y_max_i = box
        for box_k in kalmanbbox:
            x_min_k,y_min_k,x_max_k,y_max_k,id_k = box_k
            if x_min_i == x_min_k and y_min_i == y_min_k and x_max_i == x_max_k and y_max_i == y_max_k:
                tracker[str(id)] = id_k
    
    print(tracker)

def main():
    print("Load video")
    # Load the boxes from fasterRCNN
   
    frame_boxes = read_boxes_from_file(RESULTS_PATH + FILE_IN)

    cap = cv2.VideoCapture(VIDEO_PATH)

    kalman_tracker = Sort()
    kalman_tracker_dict = {}


    num_frame = 0
    for frame in frame_boxes:
        ret, frame_img = cap.read()

            # Check if frame was successfully read
        if not ret:
            break

        detected_bboxes = convert_to_array(frame)
        actual_tracks = kalman_tracker.update(detected_bboxes)

        #kalman_tracker returns the predicted bounding boxes and the id of the object
        kalman_predicted_bb = actual_tracks[:, 0:4]
        kalman_predicted_id = actual_tracks[:,4]

        x_min = {}
        y_min = {}
        x_max = {}
        y_max = {}
        for id,bbox in zip(kalman_predicted_id,kalman_predicted_bb):
            x_min[str(int(id))] = bbox[0]
            y_min[str(int(id))] = bbox[1]
            x_max[str(int(id))] = bbox[2]
            y_max[str(int(id))] = bbox[3]
        kalman_tracker_dict[str(num_frame)] = {'x_min': x_min,
                                               'y_min': y_min,
                                               'x_max': x_max,
                                               'y_max': y_max,}
        
        img_draw = frame_img.copy()
        for track in kalman_tracker.trackers:
            det = convert_x_to_bbox(track.kf.x).squeeze()
            #det, _ = track.last_detection()
            img_draw = cv2.rectangle(img_draw, (int(det[0]), int(det[1])), (int(det[2]), int(det[3])), track.visualization_color, 2)
            img_draw = cv2.rectangle(img_draw, (int(det[0]), int(det[1]-20)), (int(det[2]), int(det[1])), track.visualization_color, -2)
            img_draw = cv2.putText(img_draw, str(track.id), (int(det[0]), int(det[1])), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
            print(track.history)
            for detection in track.history:
                detection_center = ( int((detection[0][0]+detection[0][2])/2), int((detection[0][1]+detection[0][3])/2) )
                img_draw = cv2.circle(img_draw, detection_center, 5, track.visualization_color, -1)
                
        cv2.imshow('Tracking results', cv2.resize(img_draw, (int(img_draw.shape[1]*0.5), int(img_draw.shape[0]*0.5))))
        k = cv2.waitKey(1)
        if k == ord('q'):
            return
        num_frame+=1

    save_json(kalman_tracker_dict, './results/kalman.json')
    i = 0
    for track in kalman_tracker.trackers:
        print(track.id)
        print(track.history)
        print(convert_x_to_bbox(track.kf.x))
        i+=1
        if i >5: 
            break
        



if __name__ == "__main__":
    main()
    '''if SAVE or VISUALIZE:
        visualize_tracking(VIDEO_PATH, RESULTS_PATH + FILE_OUT,
                       VIDEO_OUT_PATH, SAVE, VISUALIZE)'''
