import cv2
import numpy as np
import os
import csv
import json

from sort import *
from week_utils import *

SAVE = False
VISUALIZE = True

DATASET_PATH = "../Data/aic19-track1-mtmc-train/train/S01"
VIDEO_PATH = "../Data/AICity_data/train/S03/c010/vdo.avi"
#VIDEO_OUT_PATH = "./results/kalman_tracking_vdo.avi"

RESULTS_PATH = "./results/"
FILE_IN = "bbxs_clean.json"
FILE_OUT = "kalman"

def get_centroid(box: np.ndarray):  # box [xmin,ymin,xmax,ymax]
    x_center = int((box[0][0] + box[0][2]) / 2)
    y_center = int((box[0][1] + box[0][3]) / 2)
    return x_center, y_center


class KalmanFilter:
    def __init__(self, max_age=30, iou_threshold=0.4):
        self.max_age = int(max_age)
        self.iou_threshold = iou_threshold
        self.kalman_tracker = Sort(
            max_age=int(max_age), iou_threshold=float(iou_threshold)
        )
        self.kalman_tracker_dict = {}
        self.n_frame = 0

    def next_frame(self):
        self.n_frame += 1

    def read_boxes_from_file(self, file_path):
        with open(file_path, "r") as file:
            frames_boxes = json.load(file)
        return frames_boxes
    
    def read_boxes_from_csv_file(self, file_path, num_frames):
        with open(file_path, newline='') as gtfile:
            csv_reader = csv.reader(gtfile)
            csv_boxes = {}

            # Iterate through each row in the CSV file
            for row in csv_reader:
                # Accessing each value in the row
                # Don't take into account ID
                frame, _, left, top, width, height, _, _, _, _ = map(int, row)
                frame -= 1 # starts at 1

                if frame not in csv_boxes:
                    csv_boxes[frame] = {"xmin": [], "xmax": [], "ymin": [], "ymax": [], "confidence": []}

                csv_boxes[frame]["xmin"].append(left)
                csv_boxes[frame]["ymin"].append(top)
                csv_boxes[frame]["xmax"].append(left + width)
                csv_boxes[frame]["ymax"].append(top + height)
                csv_boxes[frame]["confidence"].append(1) # GT

            frames_boxes = [{} for _ in range(num_frames)]
            for i in range(num_frames):
                if i not in csv_boxes:
                    frames_boxes[i] = {"xmin": {}, "xmax": {}, "ymin": {}, "ymax": {}, "confidence": {}}
                else:
                    frames_boxes[i]["xmin"] = {str(j): csv_boxes[i]["xmin"][j] for j in range(len(csv_boxes[i]["xmin"]))}
                    frames_boxes[i]["ymin"] = {str(j): csv_boxes[i]["ymin"][j] for j in range(len(csv_boxes[i]["ymin"]))}
                    frames_boxes[i]["xmax"] = {str(j): csv_boxes[i]["xmax"][j] for j in range(len(csv_boxes[i]["xmax"]))}
                    frames_boxes[i]["ymax"] = {str(j): csv_boxes[i]["ymax"][j] for j in range(len(csv_boxes[i]["ymax"]))}
                    frames_boxes[i]["confidence"] = {str(j): csv_boxes[i]["confidence"][j] for j in range(len(csv_boxes[i]["confidence"]))}

        return frames_boxes

    def convert_to_array(self, frame):
        x_min = frame["xmin"]
        x_max = frame["xmax"]
        y_min = frame["ymin"]
        y_max = frame["ymax"]
        score = frame["confidence"]

        boxes = []
        for id in x_min:
            box = [x_min[id], y_min[id], x_max[id], y_max[id], score[id]]
            boxes.append(np.array(box))

        return np.array(boxes)

    def update(self, detected_bbox):
        # Convert Detected BBoxes as arrays
        detected_bboxes = self.convert_to_array(detected_bbox)

        # Predict
        if len(detected_bboxes) == 0:
            detected_bboxes = np.empty((0, 5))
        actual_tracks = self.kalman_tracker.update(detected_bboxes)

        # Save
        self.update_kalman_tracker_dict(actual_tracks)

    def update_kalman_tracker_dict(self, actual_tracks):
        # Save bounding box to json
        kalman_predicted_bb, kalman_predicted_id = (
            actual_tracks[:, 0:4],
            actual_tracks[:, 4],
        )

        x_min, y_min, x_max, y_max = {}, {}, {}, {}
        for id, bbox in zip(kalman_predicted_id, kalman_predicted_bb):
            x_min[str(int(id))] = bbox[0]
            y_min[str(int(id))] = bbox[1]
            x_max[str(int(id))] = bbox[2]
            y_max[str(int(id))] = bbox[3]

        self.kalman_tracker_dict[str(self.n_frame)] = {
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
        }
        self.n_frame +=1

    def draw_tracking_result(self, frame_img):
        img_draw = frame_img.copy()
        for track in self.kalman_tracker.trackers:

            # Don't show non-updated trackers
            if track.time_since_update > 0:
                continue

            box = convert_x_to_bbox(track.kf.x).squeeze()
            # Draw Bounding Box
            img_draw = cv2.rectangle(
                img_draw,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                track.visualization_color,
                2,
            )
            # Draw Text Box
            img_draw = cv2.rectangle(
                img_draw,
                (int(box[0]), int(box[1])),
                (int(box[0] + 50), int(box[1] - 30)),
                track.visualization_color,
                -1,
            )
            # Write Text
            img_draw = cv2.putText(
                img_draw,
                str(track.id),
                (int(box[0]), int(box[1])),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                (0, 0, 0),
                2,
            )

            for detection in track.history:
                detection_center = get_centroid(detection)
                img_draw = cv2.circle(
                    img_draw, detection_center, 5, track.visualization_color, -1
                )

        return img_draw

    def execute(
        self,
        video_path: str = VIDEO_PATH,
        results_path: str = RESULTS_PATH,
        file_in: str = FILE_IN,
        file_out: str = FILE_OUT,
        generate_video: bool = True,
        vizualize: bool = True,
    ):
        cap = cv2.VideoCapture(video_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        _, extension = os.path.splitext(file_in)
        if extension.lower() == '.json':
            frame_boxes = self.read_boxes_from_file(file_in)
        elif extension.lower() in ['.txt', '.csv']:
            frame_boxes = self.read_boxes_from_csv_file(file_in, n_frames)
        else:
            print(f'Unsupported {extension} format')
            raise

        #out = None
        #if generate_video:
        #    output_video_filename = results_path + file_out + ".mp4"
        #    codec = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for AVI format
        #    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
        #    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        #    out = cv2.VideoWriter(
        #        output_video_filename, codec, fps, (frame_width, frame_height)
        #    )

        for num_frame in tqdm(range(n_frames), desc="Tracking bounding boxes"):
            ret, frame_img = cap.read()
            if not ret:
                break

            # Predict
            self.update(frame_boxes[num_frame])

            # Draw
            if vizualize or generate_video:
                img_draw = self.draw_tracking_result(frame_img)

            if vizualize:
                cv2.imshow(
                    "Tracking results",
                    cv2.resize(
                        img_draw,
                        (int(img_draw.shape[1] * 0.5), int(img_draw.shape[0] * 0.5)),
                    ),
                )
                if cv2.waitKey(1) == ord("q"):
                    break

            # if generate_video:
            #     out.write(img_draw)

        save_json(self.kalman_tracker_dict, results_path + file_out + ".json")

def main():
    for c_dir in os.listdir(DATASET_PATH):
        print(c_dir)
        c_dir_path = os.path.join(DATASET_PATH, c_dir)
        if os.path.isdir(c_dir_path):
            # Inside each 'c00x' directory
            video_file = os.path.join(c_dir_path, 'vdo.avi')
            if not os.path.isfile(video_file):
                print(f'Video file {video_file} does not exist')
                raise
            gt_file = os.path.join(c_dir_path, 'gt', 'gt.txt')
            # gt format: [frame, ID, left, top, width, height, 1, -1, -1, -1]
            # Only vehicles that pass through at least 2 cameras are taken into account. 
            if not os.path.isfile(gt_file):
                print(f'GT file {gt_file} does not exist')
                raise

            # Do kalman tracking for each video
            kalman = KalmanFilter(max_age=30, iou_threshold=0.4)

            kalman.execute(
                video_path=video_file,
                results_path=RESULTS_PATH,
                file_in=gt_file,
                file_out=FILE_OUT + '_' + c_dir,
                generate_video=False,
                vizualize=False,
            )


if __name__ == "__main__":
    main()
