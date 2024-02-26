import cv2
import numpy as np
from tqdm import tqdm


class GaussianModel:
    def __init__(
        self,
        video_path: str,
        train_split: float = 0.25,
        kernel_open_size: int = 3,
        kernel_close_size: int = 31,
        area_threshold: int = 1500,
    ) -> None:
        self.video = cv2.VideoCapture(video_path)
        self.num_frames = self.video.get(cv2.CAP_PROP_FRAME_COUNT)
        print(f"Number of frames: {self.num_frames}")

        self.train_split = train_split
        self.kernel_open_size = kernel_open_size
        self.kernel_close_size = kernel_close_size
        self.area_threshold = area_threshold

    def postprocess(self, binary: np.ndarray):
        kernel_open = np.ones((self.kernel_open_size, self.kernel_open_size))
        kernel_close = np.ones((self.kernel_close_size, self.kernel_close_size))
        postprocessed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
        postprocessed = cv2.morphologyEx(postprocessed, cv2.MORPH_CLOSE, kernel_close)
        return postprocessed

    def detect_object(self, binary: np.ndarray, plot: bool = False):
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        predictions = []
        height = binary.shape[0]
        width = binary.shape[1]

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.area_threshold:
                contour = contour[:, 0, :]
                xmin = int(np.min(contour[:, 0]))
                ymin = int(np.min(contour[:, 1]))
                xmax = int(np.max(contour[:, 0]))
                ymax = int(np.max(contour[:, 1]))

                if (xmax - xmin) < width * 0.4 and (ymax - ymin) < height * 0.4:
                    pred = {"bbox": [xmin, ymin, xmax, ymax]}
                    predictions.append(pred)
                    if plot:
                        binary_colored = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                        cv2.rectangle(binary_colored, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                        cv2.imshow("Binary", binary_colored)
                        cv2.waitKey(0)
        
        return predictions

    def segment(self, alpha: float):
        predictions = {}
        test_frames = int(self.num_frames * (1 - self.train_split))
        for _ in tqdm(range(test_frames), desc="Predicting test frames"):
            ret, frame = self.video.read()
            if not ret:
                break
            frame_id = str(int(self.video.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            foreground = np.abs(gray_frame - self.background_mean) >= alpha * (self.background_std + 2)
            foreground = 255 - (foreground * 255).astype(np.uint8)
            postprocessed_foreground = self.postprocess(foreground)
            prediction = self.detect_object(postprocessed_foreground)
            predictions.update({frame_id: prediction})

        return predictions

    def compute_mean_std(self):
        width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        train_frames = int(self.num_frames * self.train_split)
        frames = np.empty((train_frames, height, width), dtype=np.float32)
        for i in tqdm(range(train_frames), desc="Computing mean and std"):
            ret, frame = self.video.read()
            if not ret:
                break
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = gray_frame / 255.0
            frames[i] = gray_frame

        self.background_mean = np.mean(frames, axis=0)
        self.background_std = np.std(frames, axis=0)
