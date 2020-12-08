import random
from pathlib import Path

import albumentations as albu
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import get_video_len, read_img_cv2, read_video
from .transforms import HorizontalFlip, ShiftScale

class ClipDataset(Dataset):
    def __init__(self, video_path, neighbors: tuple, df, train: bool, transforms=None, only_accidents=False):
        self.video_path = Path(video_path)
        self.images_dir = self.video_path.parent.parent / 'train_images'
        self.df = df[df['video'] == self.video_path.name]
        self.train = train
        self.transforms = transforms        # Color transforms
        self.only_accidents = only_accidents

        self.min = min(neighbors)
        self.max = max(neighbors)
        self.neighbors = np.array(neighbors)

        self.flip = HorizontalFlip()
        self.shift_scale = ShiftScale(
            shift_limit=(-0.1, 0.1),
            scale_limit=(-0.5, 0.5),
            p=0.5
        )

    def __getitem__(self, index):
        index += abs(self.min)
        images, boxes, labels = self.load_images_and_boxes(index)

        if self.train:
            images, boxes, labels = self.shift_scale(*self.flip(images, boxes, labels))
        # Color augmentation for each frame independently
        if self.transforms is not None:
            for i in range(len(images)):
                images[i] = self.transforms(
                    image=images[i],
                    bboxes=boxes,
                    labels=labels,
                )['image']

        boxes = torch.stack(tuple(
            map(torch.tensor, zip(*boxes))
        )).permute(1, 0)
        boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]]  # yxyx: be warning

        target = dict(
            boxes=boxes,
            labels=torch.tensor(labels),
        )

        return torch.stack(images, dim=1), target

    def load_images_and_boxes(self, index):
        frame_indexes = self.neighbors + index
        image_names = self.df[
            self.df['frame'].isin(frame_indexes+1)
        ].image_name.unique()
        images = [read_img_cv2(self.images_dir / image_name) for image_name in image_names]

        records = self.df[self.df['frame'] == (index + 1)].copy()
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
        # return 5
        return get_video_len(self.video_path) - self.max - abs(self.min)


class ValidationClipDataset(Dataset):
    def __init__(self, video_path, neighbors: tuple, df, transforms=tuple()):
        self.video_path = Path(video_path)
        self.df = df[df['video'] == self.video_path.name]
        self.transforms = albu.Compose(list(transforms))

        self.min = min(neighbors)
        self.max = max(neighbors)
        self.neighbors = np.array(neighbors)

        video = read_video(video_path)
        self.video = torch.stack([
            self.transforms(image=img)['image'] for img in video
        ])

    def __getitem__(self, index):
        index += abs(self.min)
        frame_num = index + 1

        images, boxes = self.load_images_and_boxes(index)
        return torch.stack(images, dim=1), boxes, frame_num

    def load_images_and_boxes(self, index):
        frame_indexes = self.neighbors + index
        images = [self.video[i] for i in frame_indexes]

        records = self.df[self.df['frame'] == (index + 1)].copy()
        records = records[
            (records['impact'] > 1) &
            (records['confidence'] > 1) &
            (records['visibility'] > 0)
            ]
        boxes = records[['left', 'top', 'width', 'height']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        return images, boxes

    def __len__(self):
        return len(self.video) - self.max - abs(self.min)

