import cv2
import json
import colorsys
import os
import numpy as np


VIDEO_PATH = '../Data/AICity_data/train/S03/c010/vdo.avi'
ANNOTATIONS_PATH = './results/bbxs_clean_tracked.json'
SAVE = False


def generate_rainbow_cv2_colors(num_colors):
    # Generate evenly spaced hues across the rainbow spectrum
    hue_values = [i / num_colors for i in range(num_colors)]

    # Set a fixed saturation and value for vibrant colors
    saturation = 1.0
    value = 1.0

    # Initialize a dictionary to store BGR representations of the colors
    colors_bgr = []

    # Convert each hue to RGB and then to BGR, and store the colors in the dictionary
    for hue in hue_values:
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        bgr = tuple(int(255 * x) for x in rgb[::-1])  # Convert RGB to BGR
        colors_bgr.append(bgr)

    return colors_bgr


def visualize_tracking():
    # Open video file
    cap = cv2.VideoCapture(VIDEO_PATH)

    # Check if video file opened successfully
    if not cap.isOpened():
        print("Error opening video file")

    with open(ANNOTATIONS_PATH, 'r') as f:
        bbxs = json.load(f)

    i = 0
    colors_table = generate_rainbow_cv2_colors(20)

    # Loop through each frame of the video
    while cap.isOpened():
        # Read frame from video
        ret, frame = cap.read()

        # Check if frame was successfully read
        if not ret:
            print("Error reading frame")
            break

        img_draw = frame.copy()
        for k in bbxs[i]['track']:
            tl = (round(bbxs[i]['xmin'][k]), round(bbxs[i]['ymin'][k]))
            br = (round(bbxs[i]['xmax'][k]), round(bbxs[i]['ymax'][k]))
            img_draw = cv2.rectangle(img_draw, (tl), (br), colors_table[int(k)], 2)
            img_draw = cv2.putText(
                img_draw, k, (tl), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
            # Draw circles for previous detections
            # for detection in track.detections:
            #    detection_center = ( int((detection[0]+detection[2])/2), int((detection[1]+detection[3])/2) )
            #    img_draw = cv2.circle(img_draw, detection_center, 5, track.visualization_color, -1)

        cv2.imshow('Tracking results', cv2.resize(
            img_draw, (int(img_draw.shape[1]*0.5), int(img_draw.shape[0]*0.5))))
        k = cv2.waitKey(1)
        if k == ord('q'):
            return

        if SAVE:
            # TODO: save video of tracking
            track_exp_name = f"tracking_bbxs_clean"
            #path_to_res_folder = os.path.join('./results', track_exp_name)
            #os.makedirs(path_to_res_folder, exist_ok=True)
            #cv2.imwrite(path_to_res_folder + '/image_' + str(i).zfill(4) + '.jpg',
            #            cv2.resize(img_draw, tuple(np.int0(0.5*np.array(img_draw.shape[:2][::-1])))))

        i += 1


if __name__ == "__main__":
    visualize_tracking()
