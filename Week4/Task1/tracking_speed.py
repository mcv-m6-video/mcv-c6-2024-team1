import argparse
import json

import cv2
import numpy as np

from ViewTransformer import ViewTransformer
from sort import *
from week_utils import *
import supervision as sv
from collections import defaultdict, deque


DEFAULT_RESULTS_PATH = "./results/"
DEFAULT_FILE_IN = "bbxs_clean.json"
DEFAULT_FILE_OUT = "kalman"
DEFAULT_STOP_THRESHOLD = 10
DEFAULT_SPEED_THRESHOLD = 3

# Read gt file
VIDEO_PATH = "../../Data/AICity_data_S03_C010/AICity_data/train/S03/c010/vdo.avi"


def get_centroid(box: np.ndarray):  # box [xmin,ymin,xmax,ymax]
    x_center = int((box[0][0] + box[0][2]) / 2)
    y_center = int((box[0][1] + box[0][3]) / 2)
    return x_center, y_center


video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)


class KalmanSpeed:
    def __init__(
        self,
        max_age=30,
        iou_threshold=0.4,
        view_transformer: ViewTransformer = None,
        stop_threshold: int = DEFAULT_STOP_THRESHOLD,
        speed_threshold: int = DEFAULT_SPEED_THRESHOLD,
    ):
        self.max_age = int(max_age)
        self.iou_threshold = iou_threshold
        self.kalman_tracker = Sort(
            max_age=int(max_age), iou_threshold=float(iou_threshold)
        )
        self.kalman_tracker_dict = {}
        self.n_frame = 0
        self.view_transformer = view_transformer
        self.coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
        self.stop_threshold = stop_threshold
        self.speed_threshold = speed_threshold

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
        self.n_frame += 1

    def calculate_speed(self, tracker_id, count_speed):
        # Wait to have enough data
        if (
            len(self.coordinates[tracker_id]) > video_info.fps / 2
        ):  #  if we have at least 0.5 seconds of data
            # Calculate the speed
            coordinate_start = self.coordinates[tracker_id][-1]  # tuple
            coordinate_end = self.coordinates[tracker_id][0]  # tuple
            # compute distance between the first and last coordinates (tuples)
            distance = np.sqrt(
                (coordinate_end[0] - coordinate_start[0]) ** 2
                + (coordinate_end[1] - coordinate_start[1]) ** 2
            )
            time = len(self.coordinates[tracker_id]) / video_info.fps
            speed = distance / time * 3.6
            if speed < self.speed_threshold:
                count_speed[tracker_id] += 1

            return speed
        return None

    def draw_tracking_result(self, frame_img, count_speed, corners):
        img_draw = frame_img.copy()

        for track in self.kalman_tracker.trackers:

            # Don't show non-updated trackers
            if track.time_since_update > 0:
                continue

            # Get centroid of the bounding box
            centroid = get_centroid(convert_x_to_bbox(track.kf.x))
            inside = (
                centroid[0] > corners[0]
                and centroid[0] < corners[1]
                and centroid[1] > corners[2]
                and centroid[1] < corners[3]
            )

            # Transform centroid using ViewTransformer
            if self.view_transformer:
                # Check if centroid is within the source zone
                if inside:
                    transformed_centroid = self.view_transformer.transform_points(
                        np.array([[centroid[0], centroid[1]]])
                    )
                    centroid = tuple(transformed_centroid.squeeze())
                    self.coordinates[track.id].append(centroid)
                else:
                    # Centroid not in source region
                    pass

            # add the centroid as coordinate of the track id to calculate the speed
            self.coordinates[track.id].append(centroid)

            box = convert_x_to_bbox(track.kf.x).squeeze()
            speed = self.calculate_speed(track.id, count_speed)
            if count_speed[track.id] > self.stop_threshold or not inside:
                continue

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
            # Write Text with speed if in source zone, otherwise only write ID

            if speed is not None:
                text = f"{track.id} {speed:.2f} km/h" if speed > 0 else f"{track.id}"
                img_draw = cv2.putText(
                    img_draw,
                    text,
                    (int(box[0]), int(box[1])),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1,
                    (0, 0, 0),
                    2,
                )
            else:
                text = f"{track.id}"
                img_draw = cv2.putText(
                    img_draw,
                    text,
                    (int(box[0]), int(box[1])),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1,
                    (0, 0, 0),
                    2,
                )

            for detection in track.history:
                detection_center = get_centroid(detection)
                img_draw = cv2.circle(
                    img_draw, detection_center, 3, track.visualization_color, -1
                )

        return img_draw

    def execute(
        self,
        video_path: str = VIDEO_PATH,
        results_path: str = DEFAULT_RESULTS_PATH,
        file_in: str = DEFAULT_FILE_IN,
        file_out: str = DEFAULT_FILE_OUT,
        generate_video: bool = True,
        vizualize: bool = True,
        corners: list = None
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
        count_speed = defaultdict(int)
        while True:
            ret, frame_img = cap.read()
            if not ret:
                break

            # Predict
            self.update(frame_boxes[num_frame])
            # Draw
            img_draw = self.draw_tracking_result(frame_img, count_speed, corners)

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

        #save_json(self.kalman_tracker_dict, results_path + file_out + ".json")


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
        default=DEFAULT_RESULTS_PATH,
        help="Path to the results directory",
    )
    parser.add_argument(
        "--detections",
        type=str,
        default=DEFAULT_FILE_IN,
        help="Input json with detections",
    )
    parser.add_argument(
        "--o_name", type=str, default=DEFAULT_FILE_OUT, help="Output file name"
    )
    parser.add_argument(
        "--store",
        action="store_true",
        default=False,
        help="Flag to generate video output",
    )
    parser.add_argument(
        "--vizualize",
        action="store_true",
        default=False,
        help="Flag to generate video output",
    )
    parser.add_argument(
        "--stop-threshold",
        type=int,
        default=DEFAULT_STOP_THRESHOLD,
        help="Threshold to stop the detection when it has been stop more than `stop-threshold` times",
    )
    parser.add_argument(
        "--speed-threshold",
        type=int,
        default=DEFAULT_SPEED_THRESHOLD,
        help="Threshold to stop detecting speed",
    )

    # Kalman Filter Vars
    parser.add_argument(
        "--thr", type=float, default=0.4, help="Min IoU to keep a track"
    )
    parser.add_argument("--max_age", type=float, default=30, help="Max skip frames")

    args = parser.parse_args()

    source = np.array([[550, 150], [1100, 150], [100, 1050], [1900, 1050]])
    target = np.array([[0, 0], [15, 0], [0, 200], [15, 200]])
    # source = np.array([[510, 300], [770, 300], [40, 600], [1200, 600]])
    # target = np.array([[0, 0], [40, 0], [0, 450], [40, 450]])
    # source = np.array([[770, 720], [1000, 720], [570, 1000], [1120, 1000]])
    # target = np.array([[0, 0], [8, 0], [0, 150], [8, 150]])

    # get tl, tr, bl, br from source
    corners = [source[0][0], source[1][0], source[2][0], source[3][0]]
    vt = ViewTransformer(source, target)
    kalmanSpeedTracking = KalmanSpeed(
        max_age=args.max_age,
        iou_threshold=args.thr,
        view_transformer=vt,
        stop_threshold=args.stop_threshold,
        speed_threshold=args.speed_threshold,
    )

    kalmanSpeedTracking.execute(
        video_path=args.video_path,
        results_path=args.results_path,
        file_in=args.detections,
        file_out=args.o_name,
        generate_video=args.store,
        vizualize=args.vizualize,
        corners=corners
    )

# python3 tracking_speed.py --store --vizualize --o_name kalman_tracking_vdo --detections bbxs_clean_tracked.json
