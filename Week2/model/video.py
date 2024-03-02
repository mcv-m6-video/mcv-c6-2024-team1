import os
from typing import Iterator

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from functional import seq

class Video(Dataset):

    def __getitem__(self, index) -> Image:
        return default_loader(self.files[index])

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.files = seq(os.listdir(video_path)).map(lambda p: os.path.join(video_path, p)).sorted().to_list()

    def get_frames(self, start: int = 0, end: int = 2141) -> Iterator[Image.Image]:
        for i in range(start, end):
            yield self[i]

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.files)

    def __str__(self):
        return 'Video(path={})'.format(self.video_path)
