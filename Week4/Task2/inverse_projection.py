from pathlib import Path
import itertools
import pickle
import numpy as np
from numpy.typing import ArrayLike
import cv2
import matplotlib as mpl
import re
import torch
from typing import List
import heapq
import random
import os
import pandas as pd
import imageio
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import List
import argparse
import pickle

from syn_cam import *
from mot.tracklet import Tracklet


DISTANCE_THRESHOLD = 5
BREAK_DISTANCE = 0.001
EARTH_RADIUS = 6371001
DEG2RAD = np.pi / 180
MTMC_TRACKLETS_NAME = "mtmc_tracklets"
OUTPUT_DIR = '../results'
CSV_TRACKLETS = ['c001_tracklet.csv', 'c002_tracklet.csv',
                 'c003_tracklet.csv', 'c004_tracklet.csv', 'c005_tracklet.csv']


class MultiCameraTracklet:
    def __init__(self, new_id, tracks: List[Tracklet] = []) -> None:
        self.id = new_id
        self.tracks = tracks
        self.cams = []
        self.visualization_color = (int(
            random.random() * 256), int(random.random() * 256), int(random.random() * 256))
        for t in tracks:
            self.cams.append(t.cam)

    def add_track(self, mtrack):
        # self.cams.extend(mtrack.cams)
        self.tracks.extend(mtrack.tracks)
        for track in mtrack.tracks:
            self.cams.append(track.cam)
        self.cams = list(set(self.cams))


def multicam_track_similarity(mtrack1: MultiCameraTracklet, mtrack2: MultiCameraTracklet, linkage: str,
                              sims: np.array) -> float:
    """Compute the similarity score between two multicam tracks.

    Parameters
    ----------
    mtrack1: first multi-camera tracklet.
    mtrack2: second multi-camera tracklet.
    linkage: method to use for computing from ('average', 'single', 'complete', 'mean_feature')

    Returns
    -------
    sim_score: similarity between the tracks.
    """

    # similarity of all pairs of tracks between mtrack1 and mtrack2
    # this scales badly, but in all sensible cases multicam tracks contain only a few tracks
    all_sims = [sims[t1.track_id][t2.track_id]
                for t1 in mtrack1.tracks for t2 in mtrack2.tracks]
    if linkage == "average":
        return np.mean(all_sims)
    if linkage == "single":
        return np.max(all_sims)
    if linkage == "complete":
        return np.min(all_sims)
    raise ValueError("Invalid linkage parameter value.")


def read_corners(path):
    with open(path) as f:
        text = f.read()
    corners = np.array([[float(num) for num in line.split()]
                       for line in text.splitlines()])
    print(corners)
    return corners


def read_calibration(path):
    with open(path) as f:
        text = f.readlines()[0]
    matrix = np.array([[float(num) for num in row.split()]
                      for row in text.split(";")])
    return matrix


def get_color(number):
    """ Converts an integer number to a color """
    blue = int(number*30 % 256)
    green = int(number*103 % 256)
    red = int(number*50 % 256)

    return red, blue, green


class MultiCameraTrackScene:
    def __init__(self, tl_coords: ArrayLike | tuple[float, float], br_coords: ArrayLike | tuple[float, float], image_path: Path | str, offset: ArrayLike | tuple[float, float] = (0, 0)):
        self.tl_corner = np.array(tl_coords)
        self.br_corner = np.array(br_coords)
        offset = np.array(offset)
        self.tl_corner += offset
        self.br_corner += offset
        delta_theta = abs(tl_coords[0] - br_coords[0])
        delta_phi = abs(tl_coords[1] - br_coords[1])

        self.bg_image = cv2.imread(str(image_path))
        print(self.bg_image.shape)

        self.coords_to_pixels_factor = np.array(
            [-self.bg_image.shape[0] / delta_theta, self.bg_image.shape[1] / delta_phi])
        self.coords_to_meters_factor = np.array(
            [DEG2RAD * EARTH_RADIUS, DEG2RAD * EARTH_RADIUS * np.cos(tl_coords[0])])

        self.cameras: dict[int, np.ndarray] = {}
        self.objects: dict[int, np.ndarray] = {}

    def add_camera(self, id, calibration: np.ndarray):
        if id in self.cameras:
            raise ValueError(f"Camera with id {id} already exists.")
        self.cameras[id] = np.linalg.inv(calibration)

        for i, point in enumerate(itertools.product([0, 1920], np.linspace(200, 1080, num=40))):
            visualization.add_object(f"cam{id}b_{i}", id, point)

    def add_object(self, id: int, camera_id: int, pixel_position: ArrayLike | tuple[float, float]):
        # Pixel position in (x, y) format from top left corner

        if id in self.objects:
            raise ValueError(f"Object with id {id} already exists.")
        # print("Adding object")
        screen_coords = np.append(pixel_position, 1)
        # print(screen_coords)
        world_homo = (self.cameras[camera_id] @ screen_coords)
        # print(world_homo)
        self.objects[id] = world_homo[:2] / world_homo[2]
        # print(self.objects[id])
        return self.objects[id]

    def reset(self):
        for obj in list(self.objects.keys()):
            if type(obj) != str or "cam" not in obj:
                self.objects.pop(obj)

    def plot(self, show_image=True, show_fovs: list[int | str] = None):
        if show_fovs is None:
            show_fovs = self.cameras.keys()
        if show_image:
            image = self.bg_image.copy()
        else:
            image = np.zeros(self.bg_image.shape, dtype=np.int8)
        color = (255, 0, 0)
        for id, gps_coords in self.objects.items():
            # print(f"{gps_coords=}")
            # print((gps_coords - self.tl_corner))
            map_pos = (gps_coords - self.tl_corner) * \
                self.coords_to_pixels_factor
            # print(map_pos)
            if type(id) == str:
                if len(num := re.findall(r"cam(\d+)b", id)) > 0:
                    if int(num[0]) not in show_fovs:
                        continue
                if id[:5] == "cam0b":
                    color = (255, 0, 0)
                if id[:5] == "cam1b":
                    color = (0, 255, 0)
                if id[:5] == "cam2b":
                    color = (0, 0, 255)
                if id[:5] == "cam3b":
                    color = (0, 255, 255)
                if id[:5] == "cam4b":
                    color = (255, 0, 255)
                if id[:5] == "cam5b":
                    color = (255, 255, 0)
            elif type(id) == int:
                color = get_color(id)
            cv2.circle(image, np.int32([map_pos[0], map_pos[1]]), radius=5, color=color, thickness=-1)
        cv2.imshow("ZenithalView", image)
        cv2.waitKey(2000)


def get_bbox_center(bbox):
    # Return (x, y) from top left corner
    return (bbox[1] + bbox[2]/2, bbox[0] + bbox[3]/2)


def positional_match(track1: Tracklet, track2: Tracklet, scene: MultiCameraTrackScene):
    frame_intersect_start = max(track1.global_start, track2.global_start)
    frame_intersect_end = min(track1.global_end, track2.global_end)
    time_intersecting = True if frame_intersect_start < frame_intersect_end else False

    if time_intersecting:
        print(f"Intersecting during {frame_intersect_end - frame_intersect_start} frames.")
        frame_matches = 0
        processed_frames = 0
        print(f"{track1.frames=}")
        print(f"{track2.frames=}")
        for frame in range(frame_intersect_start, frame_intersect_end + 1):
            track1_cam_frame = (frame - track1.global_start)
            print(f"{track1_cam_frame=}")
            track2_cam_frame = (frame - track2.global_start)
            print(f"{track2_cam_frame=}")
            processed_frames += 1
            bbox1_center = get_bbox_center(track1.bboxes[track1.frames[track1_cam_frame]])
            bbox2_center = get_bbox_center(track2.bboxes[track2.frames[track2_cam_frame]])
            
            pos1 = scene.add_object(track1.track_id, track1.cam, bbox1_center)
            pos2 = scene.add_object(track2.track_id, track2.cam, bbox2_center)
            distance = np.sqrt(
                np.sum(((pos1 - pos2) * scene.coords_to_meters_factor)**2))

            if distance < DISTANCE_THRESHOLD:
                frame_matches += 1

            if distance > BREAK_DISTANCE:
                print("Too far away, breaking")
                scene.plot(show_image=False, show_fovs=[
                           track1.cam, track2.cam])
                scene.reset()
                return False

            scene.plot(show_image=False, show_fovs=[track1.cam, track2.cam])
            scene.reset()
        
        print(f"{processed_frames} have been processed.")
        if frame_matches / (frame_intersect_end - frame_intersect_start) > 0.4:
            print(f"Matched {i, j} through overlap")
            return True
        else:
            print(f"Not enough overlap between {i, j}")
            return False
    else:
        print("Non intersecting")
        first_track: Tracklet = min(track1, track2, key=lambda t: t.global_end)
        second_track: Tracklet = max(
            track1, track2, key=lambda t: t.global_end)
        # TODO: Use speed estimation part?
        last_bbx_centers = map(get_bbox_center, first_track.bboxes[-5:])
        last_positions = [scene.add_object(i, first_track.cam, center) for i, center in enumerate(last_bbx_centers)]
        last_velocities = [last_positions[i] - last_positions[i-1] for i in range(1, len(last_positions))]
        estimated_velocity = np.mean(last_velocities, axis=0)

        frame_diff = frame_intersect_start - frame_intersect_end
        estimated_position = estimated_velocity * \
            frame_diff + last_positions[-1]

        track2_start_position = scene.add_object("t2", second_track.cam, get_bbox_center(second_track.bboxes[0]))
        distance = np.sqrt(np.sum(((estimated_position - track2_start_position) * scene.coords_to_meters_factor)**2))

        scene.plot(show_image=False, show_fovs=[first_track.cam, second_track.cam])
        scene.reset()
        if distance < DISTANCE_THRESHOLD:
            print(f"Matched {i, j} through speed extrapolation")
            return True
        return False

def temporal_compatibility(t1, t2, matrix):
    for tr1 in t1.tracks:
        for tr2 in t2.tracks:
            if matrix[tr1.track_id, tr2.track_id]:
                return True

    return False


def check_cameras(t1, t2):
    for tr1 in t1.cams:
        for tr2 in t2.cams:
            if tr1 == tr2:
                return True
    return False


def get_tracks_by_cams(multicam_tracks: List[MultiCameraTracklet]) -> List[List[Tracklet]]:
    """Return multicam tracklets sorted by cameras."""
    if len(multicam_tracks) == 0:
        return []
    tracks_per_cam = [[] for _ in range(6)]
    for mtrack in multicam_tracks:
        for track in mtrack.tracks:
            track.track_id = mtrack.id
            tracks_per_cam[track.cam].append(track)
    return tracks_per_cam


def to_detections(tracklets):
    res = {
        "frame": [],
        "bbox_topleft_x": [],
        "bbox_topleft_y": [],
        "bbox_width": [],
        "bbox_height": [],
        "track_id": [],
    }
    if len(tracklets) == 0:
        return res

    for k in tracklets[0].static_attributes:
        res[k] = []
    for k in tracklets[0].dynamic_attributes:
        res[k] = []
    if tracklets[0].zones:
        res["zone"] = []

    for tracklet in tracklets:
        res["frame"].extend(tracklet.frames)
        res["track_id"].extend([tracklet.track_id] * len(tracklet.frames))
        for x, y, w, h in tracklet.bboxes:
            res["bbox_topleft_x"].append(int(x))
            res["bbox_topleft_y"].append(int(y))
            res["bbox_width"].append(int(round(w)))
            res["bbox_height"].append(int(round(h)))
        for static_f, val in tracklet.static_attributes.items():
            values = val if isinstance(val, list) else [
                val] * len(tracklet.frames)
            res[static_f].extend(values)
        for dynamic_f, val in tracklet.dynamic_attributes.items():
            res[dynamic_f].extend(val)
        if tracklet.zones:
            res["zone"].extend(tracklet.zones)

    # all columns should have the same length
    lengths = list(map(len, res.values()))
    lengths_equal = list(map(lambda l: l == lengths[0], lengths))
    if not all(lengths_equal):
        for k, v in res.items():
            print(f"Items in column {k}: {len(v)}")
        raise ValueError("Error: not all column lengths are equal.")

    # Sort by frame
    zipped_res = zip(res["frame"], res["bbox_topleft_x"], res["bbox_topleft_y"],
                     res["bbox_width"], res["bbox_height"], res["track_id"])

    # Sort the zipped res based on the "frame" values
    sorted_res = sorted(zipped_res, key=lambda x: x[0])

    # Unpack the sorted res into separate lists
    res["frame"], res["bbox_topleft_x"], res["bbox_topleft_y"], res["bbox_width"], \
        res["bbox_height"], res["track_id"] = zip(*sorted_res)

    return res


def save_tracklets_csv(tracklets, path):
    """Save tracklets as detections in a csv format (with attributes and zones)"""
    res = to_detections(tracklets)
    df = pd.DataFrame(res)
    df.to_csv(path, index=False)


def save_tracklets_csv_per_cam(multicam_tracks: List[MultiCameraTracklet], save_paths_per_cam: List[str]):
    tracks_per_cam = get_tracks_by_cams(multicam_tracks)
    for tracks, path in zip(tracks_per_cam, save_paths_per_cam):
        save_tracklets_csv(tracks, path)


STATIC_ATTRIBUTES = {
    "color": ["yellow", "orange", "green", "gray", "red", "blue", "white", "golden", "brown", "black",
              "purple", "pink"],
    "type": ["sedan", "suv", "van", "hatchback", "mpv",
             "pickup", "bus", "truck", "estate", "sportscar", "RV", "bike"],
}

DYNAMIC_ATTRIBUTES = {
    "brake_signal": ["off", "on"],
}


def get_attribute_value(name: str, value: int):
    """Get the description of an attribute, e.g. get_attribute_value('color', 5) -> 'blue'."""
    if name == "speed":
        return str(value)
    if name in STATIC_ATTRIBUTES:
        return STATIC_ATTRIBUTES[name][value]
    if name in DYNAMIC_ATTRIBUTES:
        return DYNAMIC_ATTRIBUTES[name][value]
    err = f"Invalid static or dynamic attribute name: {name}."
    raise ValueError(err)


def put_text(img_pil, text, x, y, color, font):
    draw = ImageDraw.Draw(img_pil)
    draw.text((x, y), text, (color[0], color[1], color[2],
                             255), font=font)
    return img_pil


def annotate(img_pil, id_label, attributes, tx, ty, bx, by, color, font):
    """ Put the id label and the features as text below or above of a bounding box. """

    draw = ImageDraw.Draw(img_pil, "RGBA")
    draw.rectangle([tx, ty, bx, by], outline=color, width=3)
    text = id_label

    textcoords = draw.multiline_textbbox((tx, by), text, font=font)

    txt_y = ty - (textcoords[3] - textcoords[1]) - 4

    # draw rectangle in the background
    coords = draw.multiline_textbbox((tx, txt_y), text, font=font)
    # add some padding
    textcoords = (coords[0] - 2, coords[1] - 2, coords[2] + 2, coords[3] + 2)
    draw.rectangle(textcoords, fill=color)

    # draw the text finally
    draw.multiline_text((tx, txt_y), text, (0, 0, 0), font=font)
    return img_pil


class Video:
    def __init__(self, font, fontsize=13):
        cmap = plt.get_cmap("Set2")
        self.colors = [cmap(i)[:3] for i in range(cmap.N)]
        cmap2 = plt.get_cmap("hsv")
        for i in np.linspace(0.1, 0.5, 7):
            self.colors.append(cmap2(i)[:3])
        self.HASH_Q = int(1e9 + 7)

        try:
            self.font = ImageFont.truetype(font, fontsize)
        except OSError:
            # log.error(f"Video: Font {font} cannot be loaded, using PIL default font.")
            print(
                f"Video: Font {font} cannot be loaded, using PIL default font.")
            self.font = ImageFont.load_default()
        self.frame_font = ImageFont.truetype(font, 18)
        self.frame_num = 0

    def render_tracks(self, frame, track_ids, track_bboxes, attributes):
        overlay = Image.fromarray(
            np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.uint8), "RGBA")
        for track_id, bbox, attrib in zip(track_ids, track_bboxes, attributes):
            tx, ty, w, h = bbox
            bx, by = int(tx + w), int(ty + h)
            color = self.colors[(self.HASH_Q * int(track_id)) %
                                len(self.colors)]
            color = tuple(int(i * 255) for i in color)

            overlay = annotate(overlay, str(track_id), attrib,
                               tx, ty, bx, by, color, self.font)

        mask = Image.fromarray((np.array(overlay) > 0).astype(np.uint8) * 192)
        frame_img = Image.fromarray(frame)
        frame_img.paste(overlay, mask=mask)

        put_text(frame_img, f"Frame {self.frame_num}",
                 0, 0, (255, 0, 0), self.frame_font)
        self.frame_num += 1

        return np.array(frame_img)


class FileVideo(Video):
    def __init__(self, font, save_path, fps, codec, format="FFMPEG", mode="I", fontsize=13):
        super().__init__(font, fontsize=fontsize)
        self.video = imageio.get_writer(save_path, format=format, mode=mode,
                                        fps=fps, codec=codec, macro_block_size=8)

    def update(self, frame, track_ids, bboxes, attributes):
        frame = self.render_tracks(frame, track_ids, bboxes, attributes)
        self.video.append_data(frame)

    def close(self):
        self.video.close()


def annotate_video_with_tracklets(input_path, output_path, tracklets, font="Hack-Regular.ttf",
                                  fontsize=13):
    video_in = imageio.get_reader(input_path)
    video_meta = video_in.get_meta_data()
    video_out = FileVideo(
        font, output_path, video_meta["fps"], video_meta["codec"], fontsize=fontsize)

    tracklets = sorted(tracklets, key=lambda tr: tr.frames[0])
    print(tracklets)
    active_tracks = {}
    nxt_track = 0

    for frame_idx, frame in enumerate(video_in):
        while nxt_track < len(tracklets) and tracklets[nxt_track].frames[0] == frame_idx:
            active_tracks[nxt_track] = 0
            nxt_track += 1

        track_ids, bboxes, attribs = [], [], []
        ended_tracks = []
        incr_tracks = []

        # gather info for the current frame
        for track_idx, ptr in active_tracks.items():
            track = tracklets[track_idx]

            try:
                static_refined = isinstance(
                    next(iter(track.static_attributes.values())), int)
            except StopIteration:
                static_refined = True

            if track.frames[ptr] == frame_idx:
                track_ids.append(track.track_id)
                bboxes.append(track.bboxes[ptr])

                attr = {}
                for k, v in track.static_attributes.items():
                    if static_refined:
                        attr[k] = v
                    else:
                        attr[k] = v[ptr]
                for k, v in track.dynamic_attributes.items():
                    attr[k] = v[ptr]
                attribs.append(attr)

                if ptr >= len(track.frames) - 1:
                    ended_tracks.append(track_idx)
                else:
                    incr_tracks.append(track_idx)

        for track_idx in ended_tracks:
            del active_tracks[track_idx]
        for track_idx in incr_tracks:
            active_tracks[track_idx] += 1

        video_out.update(frame, track_ids, bboxes, attribs)

    video_out.close()


def annotate_video_mtmc(video_in, video_out, multicam_tracks, cam_idx, **kwargs):
    tracks = get_tracks_by_cams(multicam_tracks)[cam_idx]
    print(len(tracks))
    annotate_video_with_tracklets(video_in, video_out, tracks, **kwargs)


if __name__ == "__main__":
    sequence_name = "S01"
    camera_names = ["c001", "c002", "c003", "c004", "c005"]

    bg_image_path = f"./Week4/VisualizationData/{sequence_name}/bg.png"
    # Write top left and bottom right GPS coordinates of the image in 2 lines
    corners_path = f"./Week4/VisualizationData/{sequence_name}/corners.txt"
    tl_corner, br_corner = read_corners(corners_path)
    visualization = MultiCameraTrackScene(tl_corner, br_corner, bg_image_path, offset=(0.0002, 0.0005))
    
    for i, name in enumerate(camera_names):
        calibration_path = f"./Data/aic19-track1-mtmc-train/train/{sequence_name}/{name}/calibration.txt"
        visualization.add_camera(i, read_calibration(calibration_path))

    all_tracks, t_compa, global_frames = syncronize_trackers(camera_names)

    # Check position coincidences
    for (i, j) in set(map(frozenset, np.argwhere(t_compa == 1))):
        track1: Tracklet = all_tracks[i]
        track2: Tracklet = all_tracks[j]
        
        if positional_match(track1, track2, visualization):
            merge_tracks(track1, track2)

    # precompute similarities between tracks
    f = torch.Tensor(np.stack([tr.mean_feature for tr in all_tracks]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    f.to(device)
    sim = torch.matmul(f, f.T).cpu().numpy()

    min_sim = 0.5
    candidates = []
    last_mod = [0] * len(all_tracks)
    timestamp = 1
    for i in range(len(all_tracks)):
        for j in range(i+1, len(all_tracks)):
            t1, t2 = all_tracks[i], all_tracks[j]
            if t_compa[i][j] and sim[i][j] > min_sim \
                    and t1.static_attributes['color'] == t2.static_attributes['color'] \
                    and t1.static_attributes['type'] != 11:
                candidates.append((-sim[i][j], timestamp, i, j))

    # sorted_candidates = np.sort(candidates,0)
    sorted_candidates = candidates
    heapq.heapify(sorted_candidates)
    mtracks = [MultiCameraTracklet(i, [t]) for i, t in enumerate(all_tracks)]
    remaining_tracks = set(range(len(all_tracks)))

    count = 0
    js = []
    while len(sorted_candidates) > 0:
        # c_sim, i, j = sorted_candidates[-1]
        # sorted_candidates = sorted_candidates[:-1]
        c_sim, t_insert, i, j = heapq.heappop(sorted_candidates)
        if c_sim > -min_sim:
            break
        if t_insert < max(last_mod[i], last_mod[j]):
            continue

        mtracks[i].add_track(mtracks[j])

        timestamp += 1
        remaining_tracks.remove(j)
        last_mod[i] = timestamp
        last_mod[j] = timestamp

        for i_other in remaining_tracks:
            if i_other == i:
                continue
            if check_cameras(mtracks[i], mtracks[i_other]) or not temporal_compatibility(mtracks[i], mtracks[i_other], t_compa):
                continue

            s = multicam_track_similarity(
                mtracks[i], mtracks[i_other], "average", sim)
            if s >= min_sim:
                heapq.heappush(sorted_candidates, (-s, timestamp, i, i_other))

        count += 1

    # drop invalidated mtracks and remove tracks with less than 2 cameras
    mtracks = [mtracks[i]
               for i in remaining_tracks if len(mtracks[i].cams) > 1]

    # reindex final tracks and finalize them
    for i, mtrack in enumerate(mtracks):
        mtrack.id = i

    # save per camera results
    csv_paths = []
    for i, csv_path in enumerate(CSV_TRACKLETS):
        mtmc_csv_path = os.path.join(
            OUTPUT_DIR, f"{i}_{os.path.split(csv_path)[1]}")
        csv_paths.append(mtmc_csv_path)

    save_tracklets_csv_per_cam(mtracks, csv_paths)

    annotate_video_mtmc('./Data/train/S03/c010/vdo.avi',
                        '../results/vdo_c010_tracklets.avi', mtracks, 0)
    annotate_video_mtmc('./Data/train/S03/c011/vdo.avi',
                        '../results/vdo_c011_tracklets.avi', mtracks, 1)
    annotate_video_mtmc('./Data/train/S03/c012/vdo.avi',
                        '../results/vdo_c012_tracklets.avi', mtracks, 2)
    annotate_video_mtmc('./Data/train/S03/c013/vdo.avi',
                        '../results/vdo_c013_tracklets.avi', mtracks, 3)
    annotate_video_mtmc('./Data/train/S03/c014/vdo.avi',
                        '../results/vdo_c014_tracklets.avi', mtracks, 4)
    annotate_video_mtmc('./Data/train/S03/c015/vdo.avi',
                        '../results/vdo_c015_tracklets.avi', mtracks, 5)
