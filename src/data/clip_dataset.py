import random
from pathlib import Path

import albumentations as albu
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import get_video_len, read_frames
from .transforms import HorizontalFlip, ShiftScale

class ClipDataset(Dataset):
    def __init__(self, video_path, neighbors: tuple, df, transforms=None, only_accidents=False):
        self.video_path = Path(video_path)
        self.df = df
        self.transforms = transforms        # Color transforms
        self.only_accidents = only_accidents

        neighbors = np.abs(neighbors)
        self.min = min(neighbors)
        self.max = max(neighbors)
        self.neighbors = np.array([-n for n in neighbors] + [0] + [n for n in neighbors])

        self.flip = HorizontalFlip()
        self.shift_scale = ShiftScale(
            shift_limit=(-0.1, 0.1),
            scale_limit=(-0.5, 0.5),
            p=0.5
        )

    def __getitem__(self, index):
        index += self.max
        images, boxes, labels = self.load_images_and_boxes(index)

        images, boxes, labels = self.shift_scale(*self.flip(images, boxes, labels))

        # Color augmentation for each frame independently
        if self.transforms is not None:
            for i in range(len(images)):
                images[i] = self.transforms(
                    image=images[i],
                )['image']

        boxes = torch.stack(tuple(
            map(torch.tensor, zip(*boxes))
        )).permute(1, 0)
        boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]]  # yxyx: be warning

        return torch.stack(images), boxes, torch.tensor(labels)

    def load_images_and_boxes(self, index):
        frame_indexes = self.neighbors + index
        images = read_frames(self.video_path, frame_indexes)

        records = self.df[
            (self.df['video'] == self.video_path.name) &
            (self.df['frame'] == (index + 1))
            ].copy()
        if self.only_accidents:
            records = records[
                (records['impact'] > 1) &
                (records['confidence'] > 1) &
                (records['visibility'] > 0)
                ]
        boxes = records[['left', 'top', 'width', 'height']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        labels = records['impact'].values
        return images, boxes, labels

    def __len__(self):
        return get_video_len(self.video_path) - self.max * 2
