from pathlib import Path
import itertools

import numpy as np
from numpy.typing import ArrayLike
import cv2
import matplotlib.cm as cm

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
        self.objects: dict[int, np.ndarray] = {"tl_corner": self.tl_corner, "br_corner": self.br_corner}
    
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
    
    def plot(self):
        print("Plotting:")
        image = self.bg_image.copy()
        color = (255, 0, 0)
        for id, gps_coords in self.objects.items():
            #print(f"{gps_coords=}")
            #print((gps_coords - self.tl_corner))
            map_pos = (gps_coords - self.tl_corner) * self.coords_to_pixels_factor
            #print(map_pos)
            if id[0] == "0":
                color = (255, 0, 0)
            if id[0] == "1":
                color = (0, 255, 0)
            if id[0] == "2":
                color = (0, 0, 255)
            if id[0] == "3":
                color = (0, 255, 255)
            if id[0] == "4":
                color = (255, 0, 255)
            cv2.circle(image, np.int32([map_pos[0], map_pos[1]]), radius=5, color=color, thickness=-1)
        cv2.imshow("ZenithalView", image)
        cv2.waitKey(0)

        

if __name__ == "__main__":
    sequence_name = "S01"
    bg_image_path = f"./Week4/VisualizationData/{sequence_name}/bg.png"
    # Write top left and bottom right GPS coordinates of the image in 2 lines
    corners_path = f"./Week4/VisualizationData/{sequence_name}/corners.txt"
    tl_corner, br_corner = read_corners(corners_path)
    visualization = PositionalMultiCameraTrack(tl_corner, br_corner, bg_image_path, offset=(0.0002, 0.0005))

    calibration_path = "./Data/aic19-track1-mtmc-train/train/S01/c001/calibration.txt"
    visualization.add_camera(0, read_calibration(calibration_path))
    calibration_path = "./Data/aic19-track1-mtmc-train/train/S01/c002/calibration.txt"
    visualization.add_camera(1, read_calibration(calibration_path))
    calibration_path = "./Data/aic19-track1-mtmc-train/train/S01/c003/calibration.txt"
    visualization.add_camera(2, read_calibration(calibration_path))
    calibration_path = "./Data/aic19-track1-mtmc-train/train/S01/c004/calibration.txt"
    visualization.add_camera(3, read_calibration(calibration_path))
    calibration_path = "./Data/aic19-track1-mtmc-train/train/S01/c005/calibration.txt"
    visualization.add_camera(4, read_calibration(calibration_path))
    #visualization.add_object(0, 0, [1000, 300])
    #visualization.add_object(1, 0, [1000, 100])
    #visualization.add_object(2, 0, [1000, 150])
    #visualization.add_object(3, 0, [1000, 200])
    #visualization.add_object(4, 0, [1000, 250])
    #visualization.add_object(5, 0, [1000, 300])
    #visualization.add_object(6, 0, [1000, 350])
    #visualization.add_object(7, 0, [1000, 400])
    #visualization.add_object(8, 0, [1000, 450])
    #visualization.add_object(9, 0, [1000, 500])
    #visualization.add_object(10, 0, [1000, 700])
    #visualization.add_object(11, 0, [1000, 800])
    #visualization.add_object(12, 0, [1000, 900])
    #visualization.add_object(20, 0, [500, 100])
    #visualization.add_object(21, 0, [500, 150])
    #visualization.add_object(22, 0, [500, 200])
    #visualization.add_object(23, 0, [500, 250])
    #visualization.add_object(24, 0, [500, 300])
    #visualization.add_object(25, 0, [500, 350])
    #visualization.add_object(26, 0, [500, 400])
    #visualization.add_object(27, 0, [500, 450])
    #visualization.add_object(28, 0, [500, 500])
    #visualization.add_object(29, 0, [500, 700])
    #visualization.add_object(30, 0, [500, 800])
    #visualization.add_object(31, 0, [500, 900])
    for i, point in enumerate(itertools.product(np.linspace(0, 1920, num=10), np.linspace(0, 1080, num=20))):
        visualization.add_object(f"0_{i}", 0, point)
        visualization.add_object(f"1_{i}", 1, point)
        visualization.add_object(f"2_{i}", 2, point)
        visualization.add_object(f"3_{i}", 3, point)
        visualization.add_object(f"4_{i}", 4, point)
    
    visualization.plot()