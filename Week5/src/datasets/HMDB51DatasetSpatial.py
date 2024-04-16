""" Dataset class for HMDB51 dataset. """

import os
from enum import Enum

from glob import glob, escape
import pandas as pd
import torch

from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2


class HMDB51DatasetSpatial(Dataset):
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
        temporal_stride: int,
        num_clips: int,
        five_crop_size: int,
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
            num_clips (int): Number of clips for inference.
            five_crop_size (int): Crop size used in FiveCrop.
        """
        self.videos_dir = videos_dir
        self.annotations_dir = annotations_dir
        self.split = split
        self.regime = regime
        self.clip_length = clip_length
        self.crop_size = crop_size
        self.temporal_stride = temporal_stride
        self.num_clips = num_clips
        self.five_crop_size = five_crop_size
        self.five_crop = v2.FiveCrop(self.five_crop_size)

        self.annotation = self._read_annotation()
        self.transform = self._create_transform()


    def _read_annotation(self) -> pd.DataFrame:
        """
        Read annotation files.

        Returns:
            pd.DataFrame: Dataframe containing video annotations.
        """
        split_suffix = "_test_split" + str(self.split.value) + ".txt"

        annotation = []
        for class_name in HMDB51DatasetSpatial.CLASS_NAMES:
            annotation_file = os.path.join(self.annotations_dir, class_name + split_suffix)
            df = pd.read_csv(annotation_file, sep=" ").dropna(axis=1, how='all') # drop empty columns
            df.columns = ['video_name', 'train_or_test']
            df = df[df.train_or_test == self.regime.value]
            df = df.rename(columns={'video_name': 'video_path'})
            df['video_path'] = os.path.join(self.videos_dir, class_name, '') + df['video_path'].replace('\.avi$', '', regex=True)
            df = df.rename(columns={'train_or_test': 'class_id'})
            df['class_id'] = HMDB51DatasetSpatial.CLASS_NAMES.index(class_name)
            annotation += [df]

        return pd.concat(annotation, ignore_index=True)


    def _create_transform(self) -> v2.Compose:
        """
        Create transform based on the dataset regime.

        Returns:
            v2.Compose: Transform for the dataset.
        """
        if self.regime == HMDB51DatasetSpatial.Regime.TRAINING:
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
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])


    def get_num_classes(self) -> int:
        """
        Get the number of classes.

        Returns:
            int: Number of classes.
        """
        return len(HMDB51DatasetSpatial.CLASS_NAMES)


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

        clips = []
        for n in range(self.num_clips):
            clip_start = n * self.clip_length * self.temporal_stride
            clip_end = clip_start + self.clip_length * self.temporal_stride
            clip_frames = frame_paths[clip_start:clip_end:self.temporal_stride]

            # Read and store each clip
            clip = torch.zeros((5, self.clip_length, 3, self.crop_size, self.crop_size), dtype=torch.float32)
            for i, frame_path in enumerate(clip_frames):
                frame = read_image(frame_path)  # (C, H, W)
                crops = self.five_crop(frame)
                for j in range(5):
                    clip[j, i] = self.transform(crops[j])

            clips.append(clip)

        # Get label from the annotation dataframe
        label = df_idx['class_id']

        return torch.stack(clips), label, video_path

        
    def collate_fn(self, batch: list) -> dict:
        """
        Collate function for creating batches. This version corrects the structure of the batched clips.

        Args:
            batch (list): List of samples, where each sample is a dictionary containing multiple clips, a label, and a video path.

        Returns:
            dict: Dictionary containing batched clips, labels, and paths.
        """
        # Create lists to hold batched data
        batched_clips = []
        batched_labels = []
        batched_paths = []

        # Process each sample in the batch
        for sample in batch:
            clips, label, path = sample

            # Apply transformations and permute dimensions for each clip (5, T, C, H, W)
            transformed_clips = [clip.permute(0, 2, 1, 3, 4) for clip in clips]

            # Store the transformed clips
            batched_clips.append(torch.stack(transformed_clips))
            batched_labels.append(label)
            batched_paths.append(path)

        # Stack all elements along the appropriate dimensions
        batched_clips = torch.stack(batched_clips)  # Should result in shape (num_videos, num_clips, C, T, H, W)
        batched_labels = torch.tensor(batched_labels)

        return dict(
            clips=batched_clips,  # (num_videos, num_clips, C, T, H, W)
            labels=batched_labels,  # (num_videos,)
            paths=batched_paths  # List of paths
        )

