from pathlib import Path
import itertools
import pickle
import numpy as np
from numpy.typing import ArrayLike
import cv2
import matplotlib as mpl
import re
import torch

from syn_cam import *
from mot.tracklet import Tracklet


DISTANCE_THRESHOLD = 5
BREAK_DISTANCE = 0.001
EARTH_RADIUS = 6371001
DEG2RAD = np.pi / 180

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
        self_id = new_id
        self.tracks = tracks


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

        self.coords_to_pixels_factor = np.array([-self.bg_image.shape[0] / delta_theta, self.bg_image.shape[1] / delta_phi])
        self.coords_to_meters_factor = np.array([DEG2RAD * EARTH_RADIUS, DEG2RAD * EARTH_RADIUS * np.cos(tl_coords[0])])

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
        #print("Adding object")
        screen_coords = np.append(pixel_position, 1)
        #print(screen_coords)
        world_homo = (self.cameras[camera_id] @ screen_coords)
        #print(world_homo)
        self.objects[id] = world_homo[:2] / world_homo[2]
        #print(self.objects[id])
        return self.objects[id]

    def reset(self):
        for obj in self.objects:
            if "cam" not in obj:
                self.objects.pop(obj)


    def plot(self, show_image = True, show_fovs: list[int | str] = None):
        if show_fovs is None:
            show_fovs = self.cameras.keys()
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
            if type(id) == str:
                num = re.match(r"\d+", id).group()
                if num not in show_fovs:
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
            if type(id) == int:
                color = np.round(mpl.colormaps["tab20"](id) * 255).astype(np.uint8)[::-1]
            cv2.circle(image, np.int32([map_pos[0], map_pos[1]]), radius=5, color=color, thickness=-1)
        cv2.imshow("ZenithalView", image)
        cv2.waitKey(1)


def get_bbox_center(bbox):
    return (bbox[1] + bbox[2]/2, bbox[0] + bbox[3]/2) # Return (x, y) from top left corner


def positional_match(track1: Tracklet, track2: Tracklet, scene: MultiCameraTrackScene):
    frame_intersect_start = max(track1.global_start, track2.global_start)
    frame_intersect_end = min(track1.global_end, track2.global_end)
    time_intersecting = True if frame_intersect_start < frame_intersect_end else False

    if time_intersecting:
        frame_matches = 0
        for frame in range(frame_intersect_start, frame_intersect_end + 1):
            track1_cam_frame = (frame - track1.global_start)
            track2_cam_frame = (frame - track2.global_start)
            if track1_cam_frame in track1.frames and track2_cam_frame in track2.frames:
                bbox1_center = get_bbox_center(track1.bboxes[track1.frames.index(track1_cam_frame)])
                bbox2_center = get_bbox_center(track2.bboxes[track2.frames.index(track2_cam_frame)])
            else:
                continue
            pos1 = scene.add_object(track1.track_id, track1.cam, bbox1_center)
            pos2 = scene.add_object(track2.track_id, track2.cam, bbox2_center)
            distance = np.sqrt(np.sum(((pos1 - pos2) * scene.coords_to_meters_factor)**2))

            if distance < DISTANCE_THRESHOLD:
                frame_matches += 1
            
            if distance > BREAK_DISTANCE:
                print("Too far away, breaking")
                scene.plot(show_image=False, show_fovs=[track1.cam, track2.cam])
                scene.reset()
                return False
            
            scene.plot(show_image=False, show_fovs=[track1.cam, track2.cam])
            scene.reset()
        
        if frame_matches / (frame_intersect_end - frame_intersect_start) > 0.8:
            return True
        else:
            return False
    else:
        first_track: Tracklet = min(track1, track2, key=lambda t: t.global_end)
        second_track: Tracklet = max(track1, track2, key=lambda t: t.global_end)
        #TODO: Use speed estimation part?
        last_bbx_centers = map(get_bbox_center, first_track.bboxes[-5:])
        last_positions = np.array(map(lambda x: scene.add_object(x, first_track.cam, x), last_bbx_centers))
        last_velocities = [last_positions[i] - last_positions[i-1] for i in range(1, len(last_positions))]
        estimated_velocity = np.mean(last_velocities, axis=0)

        frame_diff = frame_intersect_start - frame_intersect_end
        estimated_position = estimated_velocity * frame_diff + last_positions[-1]

        track2_start_position = scene.add_object(1, second_track.cam, get_bbox_center(second_track.bboxes[0]))
        distance = np.sqrt(np.sum(((estimated_position - track2_start_position) * scene.coords_to_meters_factor)**2))

        scene.reset()
        return distance < DISTANCE_THRESHOLD



def merge_tracks(track1: Tracklet, track2: Tracklet):
    pass


if __name__ == "__main__":
    sequence_name = "S03"
    camera_names = ["c010", "c011", "c012", "c013", "c014", "c015"]

    bg_image_path = f"./Week4/VisualizationData/{sequence_name}/bg.png"
    # Write top left and bottom right GPS coordinates of the image in 2 lines
    corners_path = f"./Week4/VisualizationData/{sequence_name}/corners.txt"
    tl_corner, br_corner = read_corners(corners_path)
    visualization = MultiCameraTrackScene(tl_corner, br_corner, bg_image_path, offset=(0.0002, 0.0005), show_fov=True)
    
    for i, name in enumerate(camera_names):
        calibration_path = f"./Data/aic19-track1-mtmc-train/train/{sequence_name}/{name}/calibration.txt"
        visualization.add_camera(i, read_calibration(calibration_path))

    for i, point in enumerate(itertools.product([0, 1920], np.linspace(200, 1080, num=40))):
        for j in range(len(camera_names)):
            visualization.add_object(f"cam{j}b_{i}", j, point)
            #visualization.plot(False)
    #visualization.plot(False)
    cv2.waitKey(0)
    
    all_tracks, t_compa , global_frames = syncronize_trackers(camera_names)

    # Check position coincidences
    for (i, j) in set(map(frozenset, np.argwhere(t_compa == 1))):
        track1: Tracklet = all_tracks[i]
        track2: Tracklet = all_tracks[j]
        
        if positional_match(track1, track2):
            merge_tracks(track1, track2)

     # precompute similarities between tracks
    f = torch.Tensor(np.stack([tr.mean_feature for tr in all_tracks]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    f.to(device)
    sim = torch.matmul(f, f.T).cpu().numpy()