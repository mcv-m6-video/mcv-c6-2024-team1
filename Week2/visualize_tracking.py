import colorsys
import json
import random

import cv2
from tqdm import tqdm

VIDEO_PATH = "../Data/AICity_data/train/S03/c010/vdo.avi"
ANNOTATIONS_PATH = "./results/bbxs_clean_tracked.json"
VIDEO_OUT_PATH = "./results/tracking_vdo.avi"
SAVE = True
VISUALIZE = True


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


def visualize_tracking(
    video_path, annotations_path, video_out_path, save_video=True, visualize=True
):
    # Open video file with tqdm
    cap = cv2.VideoCapture(video_path)

    with tqdm(
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Loading visualization"
    ) as pbar:
        # Check if video file opened successfully
        if not cap.isOpened():
            print("Error opening video file")
            raise

        with open(annotations_path, "r") as f:
            bbxs = json.load(f)

        if save_video:
            output_video_filename = video_out_path
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

        max_track = 0
        for bbx in bbxs:
            max_track_now = max(bbx["track"].values())
            max_track = max(max_track_now, max_track)

        i = 0
        colors_table = generate_rainbow_cv2_colors(max_track + 1)
        random.shuffle(colors_table)
        all_detections = {}

        # Loop through each frame of the video
        while cap.isOpened():
            # Read frame from video
            ret, frame = cap.read()

            # Check if frame was successfully read
            if not ret:
                if i + 1 < len(bbxs):
                    print("Error reading frame")
                break

            img_draw = frame.copy()
            for k in bbxs[i]["track"]:
                tl = (round(bbxs[i]["xmin"][k]), round(bbxs[i]["ymin"][k]))
                br = (round(bbxs[i]["xmax"][k]), round(bbxs[i]["ymax"][k]))
                img_draw = cv2.rectangle(
                    img_draw, (tl), (br), colors_table[bbxs[i]["track"][k]], 2
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
                            img_draw, center, 5, colors_table[bbxs[i]["track"][k]], -1
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
                        (int(img_draw.shape[1] * 0.5), int(img_draw.shape[0] * 0.5)),
                    ),
                )
                k = cv2.waitKey(1)
                if k == ord("q"):
                    visualize = False

            if save_video:
                out.write(img_draw)

            i += 1
            pbar.update(1)

    if save_video:
        out.release()
        print(f"Video saved at {video_out_path}")


if __name__ == "__main__":
    visualize_tracking(VIDEO_PATH, ANNOTATIONS_PATH, VIDEO_OUT_PATH, SAVE, VISUALIZE)
