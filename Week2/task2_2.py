import argparse
import json

import cv2
import numpy as np

from sort import *
from week_utils import *

SAVE = False
VISUALIZE = True

RESULTS_PATH = "./results/"
FILE_IN = "bbxs_clean.json"
FILE_OUT = "kalman"

AICITY_DATA_PATH = "AICity_data/train/S03/c010"
VIDEO_OUT_PATH = "./results/kalman_tracking_vdo"

# Read gt file
ANOTATIONS_PATH = "ai_challenge_s03_c010-full_annotation.xml"
VIDEO_PATH = "../Data/AICity_data/train/S03/c010/vdo.avi"
GENERATE_VIDEO = True


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

        frame_boxes = self.read_boxes_from_file(results_path + file_in)
        cap = cv2.VideoCapture(video_path)

        out = None
        if generate_video:
            output_video_filename = results_path + file_out + ".mp4"
            codec = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for AVI format
            fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            out = cv2.VideoWriter(
                output_video_filename, codec, fps, (frame_width, frame_height)
            )

        num_frame = 0
        while True:
            ret, frame_img = cap.read()
            if not ret:
                break

            # Predict
            self.update(frame_boxes[num_frame])
            # Draw
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

            if generate_video:
                out.write(img_draw)

            num_frame += 1

        save_json(self.kalman_tracker_dict, results_path + file_out + ".json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Task 2.2 Kalman Filter for object tracking"
    )

    # General vars
    parser.add_argument(
        "--video_path", type=str, default=VIDEO_PATH, help="Path to the input video"
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default=RESULTS_PATH,
        help="Path to the results directory",
    )
    parser.add_argument(
        "--detections", type=str, default=FILE_IN, help="Input json with detections"
    )
    parser.add_argument("--o_name", type=str, default=FILE_OUT, help="Output file name")
    parser.add_argument(
        "--store", type=bool, default=False, help="Flag to generate video output"
    )
    parser.add_argument(
        "--vizualize", type=bool, default=False, help="Flag to generate video output"
    )

    # Kalman Filter Vars
    parser.add_argument(
        "--thr", type=float, default=0.4, help="Min IoU to keep a track"
    )
    parser.add_argument("--max_age", type=float, default=30, help="Max skip frames")

    args = parser.parse_args()

    task22 = KalmanFilter(max_age=args.max_age, iou_threshold=args.thr)

    task22.execute(
        video_path=args.video_path,
        results_path=args.results_path,
        file_in=args.detections,
        file_out=args.o_name,
        generate_video=args.store,
        vizualize=args.vizualize,
    )
