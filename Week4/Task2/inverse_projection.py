from pathlib import Path
import itertools
import pickle
import numpy as np
from numpy.typing import ArrayLike
import cv2
import matplotlib.cm as cm
import torch
from typing import List
import heapq
import random
import os
import pandas as pd

from syn_cam import *
from mot.tracklet import Tracklet

MTMC_TRACKLETS_NAME = "mtmc_tracklets"
OUTPUT_DIR = '../results'
PICKLED_TRACKLETS = ['pickle_tracklet']

class MultiCameraTracklet:
    def __init__(self, new_id, tracks: List[Tracklet] = []) -> None:
        self.id = new_id
        self.tracks = tracks
        self.cams = []
        self.visualization_color = (int(random.random() * 256), int(random.random() * 256), int(random.random() * 256))
        for t in tracks:
            self.cams.append(t.cam)

    def add_track(self, mtrack):
        # self.cams.extend(mtrack.cams)
        self.tracks.extend(mtrack.tracks)
        for track in mtrack.tracks:
            self.cams.append(track.cam)
        self.cams = list(set(self.cams))
        self.n_cams = len(self.cams)


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
    all_sims = [sims[t1.idx][t2.idx]
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


class PositionalMultiCameraTrack:
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

        self.cameras: dict[int, np.ndarray] = {}
        self.objects: dict[int, np.ndarray] = {}

    def add_camera(self, id, calibration: np.ndarray):
        if id in self.cameras:
            raise ValueError(f"Camera with id {id} already exists.")
        self.cameras[id] = np.linalg.inv(calibration)

    def add_object(self, id: int, camera_id: int, pixel_position: ArrayLike | tuple[float, float]):
        if id in self.objects:
            raise ValueError(f"Object with id {id} already exists.")
        # print("Adding object")
        screen_coords = np.append(pixel_position, 1)
        # print(screen_coords)
        world_homo = (self.cameras[camera_id] @ screen_coords)
        # print(world_homo)
        self.objects[id] = world_homo[:2] / world_homo[2]
        # print(self.objects[id])

    def plot(self, show_image=True):
        print("Plotting:")
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
            cv2.circle(image, np.int32(
                [map_pos[0], map_pos[1]]), radius=5, color=color, thickness=-1)
        cv2.imshow("ZenithalView", image)
        cv2.waitKey(10)


def temporal_compatibility(t1, t2, matrix):
    for tr1 in t1.tracks:
        for tr2 in t2.tracks:
            if matrix[tr1.idx, tr2.idx]:
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
    tracks_per_cam = [[] for _ in range(len(multicam_tracks[0].cams))]
    for mtrack in multicam_tracks:
        for track in mtrack.tracks:
            tracks_per_cam[track.cam].append(track)
    return tracks_per_cam

def save_tracklets(tracklets, path, max_features=None):
    """Saves tracklets using pickle (with re-id features)"""
    if max_features is not None:
        for tracklet in tracklets:
            tracklet.cluster_features(max_features)
    with open(path, "wb") as fp:
        pickle.dump(tracklets, fp, protocol=pickle.HIGHEST_PROTOCOL)

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
        for x, y, w, h in tracklet.bboxes:
            res["bbox_topleft_x"].append(int(x))
            res["bbox_topleft_y"].append(int(y))
            res["bbox_width"].append(int(round(w)))
            res["bbox_height"].append(int(round(h)))
        res["track_id"].extend([tracklet.track_id] * len(tracklet.frames))
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

    return res

def save_tracklets_csv(tracklets, path):
    """Save tracklets as detections in a csv format (with attributes and zones)"""
    res = to_detections(tracklets)
    df = pd.DataFrame(res)
    df.to_csv(path, index=False)


def save_tracklets_txt(tracklets, path):
    """Save tracklets as detections in the MOTChallenge format"""
    res = to_detections(tracklets)
    res["frame"] = list(map(lambda x: x + 1, res["frame"]))
    df = pd.DataFrame(res)
    df = df[["frame", "track_id", "bbox_topleft_x", "bbox_topleft_y", "bbox_width", "bbox_height"]]
    df["conf"] = 1
    df["x"] = -1
    df["y"] = -1
    df["z"] = -1
    df.to_csv(path, index=False, header=False)

def save_tracklets_per_cam(multicam_tracks: List[MultiCameraTracklet], save_paths_per_cam: List[str]):
    tracks_per_cam = get_tracks_by_cams(multicam_tracks)
    for tracks, path in zip(tracks_per_cam, save_paths_per_cam):
        save_tracklets(tracks, path)


def save_tracklets_csv_per_cam(multicam_tracks: List[MultiCameraTracklet], save_paths_per_cam: List[str]):
    tracks_per_cam = get_tracks_by_cams(multicam_tracks)
    for tracks, path in zip(tracks_per_cam, save_paths_per_cam):
        save_tracklets_csv(tracks, path)


def save_tracklets_txt_per_cam(multicam_tracks: List[MultiCameraTracklet], save_paths_per_cam: List[str]):
    tracks_per_cam = get_tracks_by_cams(multicam_tracks)
    for tracks, path in zip(tracks_per_cam, save_paths_per_cam):
        save_tracklets_txt(tracks, path)


if __name__ == "__main__":
    sequence_name = "S03"
    camera_names = ["c010", "c011", "c012", "c013", "c014", "c015"]

    bg_image_path = f"../VisualizationData/{sequence_name}/bg.png"
    # Write top left and bottom right GPS coordinates of the image in 2 lines
    corners_path = f"../VisualizationData/{sequence_name}/corners.txt"
    tl_corner, br_corner = read_corners(corners_path)
    visualization = PositionalMultiCameraTrack(
        tl_corner, br_corner, bg_image_path, offset=(0.0002, 0.0005))

    for i, name in enumerate(camera_names):
        calibration_path = f"./Data/train/{sequence_name}/{name}/calibration.txt"
        visualization.add_camera(i, read_calibration(calibration_path))

    for i, point in enumerate(itertools.product([0, 1920], np.linspace(200, 1920, num=40))):
        for j in range(len(camera_names)):
            visualization.add_object(f"cam{j}b_{i}", j, point)
            # visualization.plot(False)
    # visualization.plot(False)
    # cv2.waitKey(0)

    all_tracks, t_compa, global_frames = syncronize_trackers()

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
            if t_compa[i][j] and sim[i][j] > min_sim and t1.static_attributes['color'] == t2.static_attributes['color']:
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

            s = multicam_track_similarity(mtracks[i], mtracks[i_other], "average", sim)
            if s >= min_sim:
                heapq.heappush(sorted_candidates, (-s, timestamp, i, i_other))

        count += 1

    # drop invalidated mtracks and remove tracks with less than 2 cameras
    mtracks = [mtracks[i] for i in remaining_tracks if len(mtracks[i].cams) > 1]

    # reindex final tracks and finalize them
    for i, mtrack in enumerate(mtracks):
        mtrack.id = i

    # save per camera results
    pkl_paths = []
    for i, pkl_path in enumerate(PICKLED_TRACKLETS):
        mtmc_pkl_path = os.path.join(OUTPUT_DIR, f"{i}_{os.path.split(pkl_path)[1]}")
        pkl_paths.append(mtmc_pkl_path)
    csv_paths = [pth.split(".")[0] + ".csv" for pth in pkl_paths]
    txt_paths = [pth.split(".")[0] + ".txt" for pth in pkl_paths]

    save_tracklets_per_cam(mtracks, pkl_paths)
    save_tracklets_csv_per_cam(mtracks, csv_paths)
    save_tracklets_txt_per_cam(mtracks, txt_paths)

    print(len(mtracks))
    print(len(all_tracks))
