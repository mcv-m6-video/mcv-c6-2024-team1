""" Dataset class for HMDB51 dataset. """

import os
import random
from enum import Enum

from glob import glob, escape
import pandas as pd
import torch

from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2


class TSNDataset(Dataset):
    """
    Dataset class for HMDB51 dataset.
    """

    class Split(Enum):
        """
        Enum class for dataset splits.
        """
        TEST_ON_SPLIT_1 = 1
        TEST_ON_SPLIT_2 = 2
        TEST_ON_SPLIT_3 = 3

    class Regime(Enum):
        """
        Enum class for dataset regimes.
        """
        TRAINING = 1
        TESTING = 2
        VALIDATION = 3

    CLASS_NAMES = [
        "brush_hair", "catch", "clap", "climb_stairs", "draw_sword", "drink", 
        "fall_floor", "flic_flac", "handstand", "hug", "kick", "kiss", "pick", 
        "pullup", "push", "ride_bike", "run", "shoot_ball", "shoot_gun", "situp", 
        "smoke", "stand", "sword", "talk", "turn", "wave", 
        "cartwheel", "chew", "climb", "dive", "dribble", "eat", "fencing", 
        "golf", "hit", "jump", "kick_ball", "laugh", "pour", "punch", "pushup", 
        "ride_horse", "shake_hands", "shoot_bow", "sit", "smile", "somersault", 
        "swing_baseball", "sword_exercise", "throw", "walk"
    ]


    def __init__(
        self, 
        videos_dir: str, 
        annotations_dir: str, 
        split: Split, 
        regime: Regime, 
        clip_length: int, 
        crop_size: int, 
        temporal_stride: int
    ) -> None:
        """
        Initialize HMDB51 dataset.

        Args:
            videos_dir (str): Directory containing video files.
            annotations_dir (str): Directory containing annotation files.
            split (Split): Dataset split (TEST_ON_SPLIT_1, TEST_ON_SPLIT_2, TEST_ON_SPLIT_3).
            regime (Regimes): Dataset regime (TRAINING, TESTING, VALIDATION).
            split (Splits): Dataset split (TEST_ON_SPLIT_1, TEST_ON_SPLIT_2, TEST_ON_SPLIT_3).
            clip_length (int): Number of frames of the clips.
            crop_size (int): Size of spatial crops (squares).
            temporal_stride (int): Receptive field of the model will be (clip_length * temporal_stride) / FPS.
        """
        self.videos_dir = videos_dir
        self.annotations_dir = annotations_dir
        self.split = split
        self.regime = regime
        self.clip_length = clip_length
        self.crop_size = crop_size
        self.temporal_stride = temporal_stride

        self.annotation = self._read_annotation()
        self.transform = self._create_transform()

        self.num_views = 3  # You can adjust this number as needed


    def _read_annotation(self) -> pd.DataFrame:
        """
        Read annotation files.

        Returns:
            pd.DataFrame: Dataframe containing video annotations.
        """
        split_suffix = "_test_split" + str(self.split.value) + ".txt"

        annotation = []
        for class_name in TSNDataset.CLASS_NAMES:
            annotation_file = os.path.join(self.annotations_dir, class_name + split_suffix)
            df = pd.read_csv(annotation_file, sep=" ").dropna(axis=1, how='all') # drop empty columns
            df.columns = ['video_name', 'train_or_test']
            df = df[df.train_or_test == self.regime.value]
            df = df.rename(columns={'video_name': 'video_path'})
            df['video_path'] = os.path.join(self.videos_dir, class_name, '') + df['video_path'].replace('\.avi$', '', regex=True)
            df = df.rename(columns={'train_or_test': 'class_id'})
            df['class_id'] = TSNDataset.CLASS_NAMES.index(class_name)
            annotation += [df]

        return pd.concat(annotation, ignore_index=True)


    def _create_transform(self) -> v2.Compose:
        """
        Create transform based on the dataset regime.

        Returns:
            v2.Compose: Transform for the dataset.
        """
        if self.regime == TSNDataset.Regime.TRAINING:
            return v2.Compose([
                v2.RandomResizedCrop(self.crop_size),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return v2.Compose([
                v2.Resize(self.crop_size), # Shortest side of the frame to be resized to the given size
                v2.CenterCrop(self.crop_size),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


    def get_num_classes(self) -> int:
        """
        Get the number of classes.

        Returns:
            int: Number of classes.
        """
        return len(TSNDataset.CLASS_NAMES)


    def __len__(self) -> int:
        """
        Get the length (number of videos) of the dataset.

        Returns:
            int: Length (number of videos) of the dataset.
        """
        return len(self.annotation)


    def __getitem__(self, idx: int) -> tuple:
        """
        Get item (video) from the dataset.

        Args:
            idx (int): Index of the item (video).

        Returns:
            tuple: Tuple containing video, label, and video path.
        """
        df_idx = self.annotation.iloc[idx]

        # Get video path from the annotation dataframe and check if it exists
        video_path = df_idx['video_path']
        assert os.path.exists(video_path)

        # Read frames' paths from the video
        frame_paths = sorted(glob(os.path.join(escape(video_path), "*.jpg"))) # get sorted frame paths
        video_len = len(frame_paths)

        # Define number of segments and segment length
        num_segments = self.num_views
        segment_length = video_len // num_segments

        # Initialize list to store sampled frames
        sampled_frames = []
        labels = []

        # Sample clips/views from each segment
        for i in range(num_segments):
            segment_start = i * segment_length
            segment_end = min(segment_start + segment_length, video_len)
            
            # Randomly select a frame within the segment
            view_frame_index = random.randint(segment_start, segment_end - self.clip_length * self.temporal_stride)

            # Sample frames from the selected frame
            segment_frames = frame_paths[view_frame_index:view_frame_index + self.clip_length * self.temporal_stride:self.temporal_stride]
            sampled_frames.extend(segment_frames)
            labels.append(df_idx['class_id'])

        # Read frames from sampled paths
        video = torch.stack([read_image(path) for path in sampled_frames])

        return video, labels, video_path

    
    def collate_fn(self, batch: list) -> dict:
        """
        Collate function for creating batches.

        Args:
            batch (list): List of samples.

        Returns:
            dict: Dictionary containing batched clips, labels, and paths.
        """
        # Unzip the batch
        unbatched_clips, unbatched_labels, paths = zip(*batch)

        # Initialize lists to store batched clips and labels
        batched_clips = []
        batched_labels = []

        # Iterate over each sample in the batch
        for i in range(len(unbatched_clips[0])):
            # Gather clips and labels for the current view
            clips = [sample[i] for sample in unbatched_clips]
            labels = [label for label in unbatched_labels]

            # Apply transformation and permute dimensions: (T, C, H, W) -> (C, T, H, W)
            transformed_clips = [self.transform(clip).permute(1, 0, 2, 3) for clip in clips]

            # Stack clips along the batch dimension: (C, T, H, W)
            batched_clips.append(torch.stack(transformed_clips))
            batched_labels.append(labels)

        return dict(
            clips=batched_clips,  # List of tensors: [(B, C, T, H, W), (B, C, T, H, W), ...]
            labels=batched_labels,  # List of lists: [[label1, label2, ...], [label1, label2, ...], ...]
            paths=paths  # List of paths: [path1, path2, ...]
        )
