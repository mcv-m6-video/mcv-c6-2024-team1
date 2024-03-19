import numpy as np
from numpy.typing import ArrayLike

def read_calibration(path):
    with open(path) as f:
        text = f.read()
    matrix = np.array([[float(num) for num in row.split()] for row in text.split(";")])
    return matrix

class ZenithalVisualization:
    def __init__(self, center: ArrayLike | tuple[float, float]):
        self.center = np.array(center)
        self.cameras: dict[int, np.ndarray] = []
        self.objects: dict[int, np.ndarray] = []
    
    def add_camera(self, id, calibration: np.ndarray):
        if id in self.cameras:
            raise ValueError(f"Camera with id {id} already exists.")
        self.cameras[id] = calibration
    
    def add_object(self, id: int, camera_id: int, pixel_position: ArrayLike | tuple[float, float]):
        if id in self.objects:
            raise ValueError(f"Object with id {id} already exists.")
        calibration_mat = self.cameras[camera_id]
        screen_to_world_mat = np.linalg.inv(calibration_mat)

        screen_coords = np.append(pixel_position, 1).T
        world_homo = (screen_to_world_mat @ screen_coords).T
        self.objects[id] = world_homo[:2] / world_homo[2]
    
    def plot(self):
        x_min = min(self.objects, key=lambda k: self.objects.get(k)[0])
        y_min = min(self.objects, key=lambda k: self.objects.get(k)[1])
        x_max = max(self.objects, key=lambda k: self.objects.get(k)[0])
        y_max = max(self.objects, key=lambda k: self.objects.get(k)[1])
        

        

if __name__ == "__main__":
    calibration_path = "./Data/aic19-track1-mtmc-train/train/S01/c001/calibration.txt"
    print(read_calibration(calibration_path))