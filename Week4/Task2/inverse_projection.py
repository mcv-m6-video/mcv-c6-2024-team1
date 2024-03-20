from pathlib import Path
import itertools
import pickle
import numpy as np
from numpy.typing import ArrayLike
import cv2
import matplotlib.cm as cm
import torch
from typing import List

from syn_cam import *
from mot.tracklet import Tracklet

def multicam_track_similarity(mtrack1, mtrack2 , linkage: str,
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
    all_sims = [sims[t1.idx][t2.idx] for t1 in mtrack1 for t2 in mtrack2]
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
    corners = np.array([[float(num) for num in line.split()] for line in text.splitlines()])
    print(corners)
    return corners

def read_calibration(path):
    with open(path) as f:
        text = f.readlines()[0]
    matrix = np.array([[float(num) for num in row.split()] for row in text.split(";")])
    return matrix

class MultiCameraTracklet:
    def __init__(self, new_id, tracks: list[int | str] = []) -> None:
        self.id = new_id
        self.tracks = tracks
        self.cams = []
        for t in tracks:
            self.cams.append(t.cam)
    
    def add_track(self,mtrack):
        self.cams.append(mtrack.cams)
        self.tracks.append(mtrack.tracks)
    



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

        self.coords_to_pixels_factor = np.array([-self.bg_image.shape[0] / delta_theta, self.bg_image.shape[1] / delta_phi])

        self.cameras: dict[int, np.ndarray] = {}
        self.objects: dict[int, np.ndarray] = {}
    
    def add_camera(self, id, calibration: np.ndarray):
        if id in self.cameras:
            raise ValueError(f"Camera with id {id} already exists.")
        self.cameras[id] = np.linalg.inv(calibration)
    


    def add_object(self, id: int, camera_id: int, pixel_position: ArrayLike | tuple[float, float]):
        if id in self.objects:
            raise ValueError(f"Object with id {id} already exists.")
        #print("Adding object")
        screen_coords = np.append(pixel_position, 1)
        #print(screen_coords)
        world_homo = (self.cameras[camera_id] @ screen_coords)
        #print(world_homo)
        self.objects[id] = world_homo[:2] / world_homo[2]
        #print(self.objects[id])
    

    
    def plot(self, show_image = True):
        print("Plotting:")
        if show_image:
            image = self.bg_image.copy()
        else:
            image = np.zeros(self.bg_image.shape, dtype=np.int8)
        color = (255, 0, 0)
        for id, gps_coords in self.objects.items():
            #print(f"{gps_coords=}")
            #print((gps_coords - self.tl_corner))
            map_pos = (gps_coords - self.tl_corner) * self.coords_to_pixels_factor
            #print(map_pos)
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
            cv2.circle(image, np.int32([map_pos[0], map_pos[1]]), radius=5, color=color, thickness=-1)
        cv2.imshow("ZenithalView", image)
        cv2.waitKey(10)

        
def temporal_compatibility(t1,t2, matrix):
    for tr1 in t1.tracks:
        for tr2 in t2.tracks:
            if matrix[tr1.idx,tr2.idx]:
                return True

    return False

def check_cameras(t1,t2):
    for tr1 in t1.cams:
        for tr2 in t2.cams:
            if tr1==tr2:
                return True
    return False
    
if __name__ == "__main__":
    sequence_name = "S03"
    camera_names = ["c010", "c011", "c012", "c013", "c014", "c015"]

    bg_image_path = f"../VisualizationData/{sequence_name}/bg.png"
    # Write top left and bottom right GPS coordinates of the image in 2 lines
    corners_path = f"../VisualizationData/{sequence_name}/corners.txt"
    tl_corner, br_corner = read_corners(corners_path)
    visualization = PositionalMultiCameraTrack(tl_corner, br_corner, bg_image_path, offset=(0.0002, 0.0005))
    
    for i, name in enumerate(camera_names):
        calibration_path = f"./Data/train/{sequence_name}/{name}/calibration.txt"
        visualization.add_camera(i, read_calibration(calibration_path))

    for i, point in enumerate(itertools.product([0, 1920], np.linspace(200, 1920, num=40))):
        for j in range(len(camera_names)):
            visualization.add_object(f"cam{j}b_{i}", j, point)
            #visualization.plot(False)
    #visualization.plot(False)
    #cv2.waitKey(0)
    
    all_tracks, t_compa , global_frames = syncronize_trackers()

     # precompute similarities between tracks
    f = torch.Tensor(np.stack([tr.mean_feature for tr in all_tracks]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    f.to(device)
    sim = torch.matmul(f, f.T).cpu().numpy()
    
    min_sim = 0.5
    candidates = []
    for i in range(len(all_tracks)):

        for j in range(i+1, len(all_tracks)):
            t1, t2 = all_tracks[i], all_tracks[j]
            if t_compa[i][j] and sim[i][j]>min_sim and t1.static_attributes['color'] == t2.static_attributes['color']:
                candidates.append([-sim[i][j],int(i),int(j)])

    sorted_candidates = np.sort(candidates,0)
    n = len(all_tracks)
    mtracks = [MultiCameraTracklet(i,[t]) for i,t in enumerate(all_tracks)]
    remaining_tracks = set(range(n))
    while len(sorted_candidates)>0:
        c_sim, i, j = sorted_candidates[-1]
        np.delete(sorted_candidates, -1)
        if c_sim>-min_sim:
            break
        mtracks[int(i)].add_track(mtracks[int(j)])
        i = int(i)
        j = int(j)
        remaining_tracks.remove(j)
        for i_other in remaining_tracks:
            if i_other == i:
                continue
            if check_cameras(mtracks[i], mtracks[i_other]) or not temporal_compatibility(mtracks[i],mtracks[i_other],t_compa):
                continue

            print(mtracks[i].tracks)
            print(i_other)
            print(mtracks[i_other].tracks)
            
            s = multicam_track_similarity(mtracks[i].tracks, mtracks[i_other].tracks,"average", sim)
            if s >= min_sim:
                sorted_candidates.append([-s, i, i_other])
                sorted_candidates = np.sort(sorted_candidates,0)
    
    # drop invalidated mtracks
    mtracks = [mtracks[i] for i in remaining_tracks]

    # reindex final tracks and finalize them
    for i, mtrack in enumerate(mtracks):
        mtrack.id = i

    print(mtracks)








