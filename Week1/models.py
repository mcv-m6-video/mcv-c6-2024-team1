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
        """
        Initialize the GaussianModel.

        Args:
            video_path (str): Path to the video file.
            train_split (float): Ratio of frames to use for training.
            kernel_open_size (int): Size of the kernel for opening operation.
            kernel_close_size (int): Size of the kernel for closing operation.
            area_threshold (int): Minimum area to consider as an object.
        """
        self.video = cv2.VideoCapture(video_path)
        self.num_frames = self.video.get(cv2.CAP_PROP_FRAME_COUNT)
        print(f"Number of frames: {self.num_frames}")

        self.train_split = train_split
        self.kernel_open_size = kernel_open_size
        self.kernel_close_size = kernel_close_size
        self.area_threshold = area_threshold

    def postprocess(self, binary: np.ndarray):
        """
        Apply morphological operations to post-process binary image.

        Args:
            binary (np.ndarray): Binary image to be post-processed.

        Returns:
            np.ndarray: Post-processed binary image.
        """
        kernel_open = np.ones((self.kernel_open_size, self.kernel_open_size))
        kernel_close = np.ones((self.kernel_close_size, self.kernel_close_size))
        postprocessed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
        postprocessed = cv2.morphologyEx(postprocessed, cv2.MORPH_CLOSE, kernel_close)
        return postprocessed

    def detect_object(self, binary: np.ndarray, plot: bool = False):
        """
        Detect objects in the binary image.

        Args:
            binary (np.ndarray): Binary image containing objects.
            plot (bool): Whether to plot detected objects.

        Returns:
            list: List of dictionaries containing bounding boxes of detected objects.
        """
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
        """
        Segment objects in the video frames.

        Args:
            alpha (float): Alpha value for segmentation.

        Returns:
            dict, list: Predictions containing bounding boxes and segmented frames.
        """
        frames = []
        predictions = {}
        test_frames = int(self.num_frames * (1 - self.train_split))
        for _ in tqdm(range(test_frames), desc="Predicting test frames"):
            ret, frame = self.video.read()
            if not ret:
                break
            frame_id = str(int(self.video.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            foreground = np.abs(gray_frame - self.background_mean) >= alpha * (self.background_std + 2)
            foreground = (foreground * 255).astype(np.uint8)
            postprocessed_foreground = self.postprocess(foreground)
            prediction = self.detect_object(postprocessed_foreground)
            predictions.update({frame_id: prediction})
            frames.append(cv2.cvtColor(postprocessed_foreground, cv2.COLOR_GRAY2RGB))

        return predictions, frames

    def compute_mean_std(self):
        """
        Compute mean and standard deviation of background frames.
        """
        width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        train_frames = int(self.num_frames * self.train_split)
        frames = np.empty((train_frames, height, width), dtype=np.float32)
        for i in tqdm(range(train_frames), desc="Computing mean and std"):
            ret, frame = self.video.read()
            if not ret:
                break
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames[i] = gray_frame

        self.background_mean = np.mean(frames, axis=0)
        self.background_std = np.std(frames, axis=0)
        print()

class AdaptativeGaussianModel(GaussianModel):
    def __init__(
        self,
        video_path: str,
        train_split: float = 0.25,
        kernel_open_size: int = 3,
        kernel_close_size: int = 31,
        area_threshold: int = 1500,
        rho: float = 0.4,
    ) -> None:
        super().__init__(
            video_path, train_split, kernel_open_size, kernel_close_size, area_threshold
        )
        self.rho = rho
    
    def updateBackground(self, foreground:np.ndarray, frame: np.ndarray):
        updated_mean = self.rho * frame + (1 - self.rho) * self.background_mean
        updated_var = self.rho * (frame - self.background_mean) ** 2 + (1 - self.rho) * self.background_std**2
        updated_std = np.sqrt(updated_var)

        self.background_mean = np.where(foreground == 0, updated_mean, self.background_mean)
        self.background_std = np.where(foreground == 0, updated_std, self.background_std)

    def segment(self, alpha: float):
        frames = []
        predictions = {}
        test_frames = int(self.num_frames * (1 - self.train_split))
        for _ in tqdm(range(test_frames), desc="Predicting test frames"):
            ret, frame = self.video.read()
            if not ret:
                break
            frame_id = str(int(self.video.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            foreground = np.abs(gray_frame - self.background_mean) >= alpha * (self.background_std + 2)
            foreground = (foreground * 255).astype(np.uint8)
            self.updateBackground(foreground, gray_frame)
            postprocessed_foreground = self.postprocess(foreground)
            prediction = self.detect_object(postprocessed_foreground)
            predictions.update({frame_id: prediction})
            frames.append(cv2.cvtColor(postprocessed_foreground, cv2.COLOR_GRAY2RGB))

        return predictions, frames

class GaussianMixtureModel(GaussianModel):
    def __init__(
        self,
        video_path: str,
        train_split: float = 0.25,
        kernel_open_size: int = 3,
        kernel_close_size: int = 31,
        area_threshold: int = 1500,
    ) -> None:
        super().__init__(
            video_path, train_split, kernel_open_size, kernel_close_size, area_threshold
        )

    def segment(self, alpha: float):
        """
        Segment objects in the video frames.

        Args:
            alpha (float): Alpha value for segmentation.

        Returns:
            dict, list: Predictions containing bounding boxes and segmented frames.
        """
        frames = []
        predictions = {}
        test_frames = int(self.num_frames * (1 - self.train_split))
        for _ in tqdm(range(test_frames), desc="Predicting test frames"):
            ret, frame = self.video.read()
            if not ret:
                break
            frame_id = str(int(self.video.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            foreground_channels = np.abs(frame - self.background_mean) >= alpha * (self.background_std + 2)
            foreground = np.any(foreground_channels, axis=2)
            foreground = (foreground * 255).astype(np.uint8)
            postprocessed_foreground = self.postprocess(foreground)
            prediction = self.detect_object(postprocessed_foreground)
            predictions.update({frame_id: prediction})
            frames.append(cv2.cvtColor(postprocessed_foreground, cv2.COLOR_GRAY2RGB))

        return predictions, frames

    def compute_mean_std(self):
        """
        Compute mean and standard deviation of background frames.
        """
        width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        train_frames = int(self.num_frames * self.train_split)
        #frames = np.empty((train_frames, height, width, 3), dtype=np.float32)
        batch = 20
        cum_n = 0
        cum_mean = 0
        cum_var = 0
        f_len = min(batch, train_frames)
        frames = np.empty((f_len, height, width, 3), dtype=np.float32)
        for i in tqdm(range(train_frames), desc="Computing mean and std"):
            ret, frame = self.video.read()
            if not ret:
                break
            color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            frames[i % batch] = color_frame
            if i % batch == batch-1:
                f_mean = np.mean(frames, axis=0)
                f_var = np.var(frames, axis=0)

                combined_n = cum_n + f_len
                combined_mean = (cum_n * cum_mean + f_len * f_mean)/(combined_n)
                combined_var = (cum_n * cum_var + f_len * f_var)/(combined_n) + cum_n * f_len * (cum_mean - f_mean)**2 / combined_n**2

                cum_n, cum_mean, cum_var = combined_n, combined_mean, combined_var

                f_len = min(batch, train_frames - i - 1)
                frames = np.empty((f_len, height, width, 3), dtype=np.float32)
        if f_len > 0:
            f_mean = np.mean(frames, axis=0)
            f_var = np.var(frames, axis=0)

            combined_n = cum_n + f_len
            combined_mean = (cum_n * cum_mean + f_len * f_mean)/(combined_n)
            combined_var = (cum_n * cum_var + f_len * f_var)/(combined_n) + cum_n * f_len * (cum_mean - f_mean)**2 / combined_n**2

            cum_n, cum_mean, cum_var = combined_n, combined_mean, combined_var
        #self.background_mean = np.mean(frames, axis=0)
        #self.background_std = np.std(frames, axis=0)
        self.background_mean = cum_mean
        self.background_std = np.sqrt(cum_var)
