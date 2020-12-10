from pathlib import Path

import albumentations as albu
import torch
import numpy as np
from torch.utils.data import Dataset

from .utils import get_video_len, read_img_cv2, read_video


class ImageDataset(Dataset):
    def __init__(self, root, df, only_accidents, transforms):
        self.images_dir = root / 'train_images'
        self.df = df
        self.image_names = list(df['image_name'].unique())
        self.transforms = transforms
        self.only_accidents = only_accidents

    def __getitem__(self, index):
        image, boxes, labels = self.load_image_and_boxes(index)

        for _ in range(100):
            sample = self.transforms(image=image, bboxes=boxes, labels=labels)

            if len(sample['bboxes']) > 0:
                image = sample['image']
                boxes = sample['bboxes']
                labels = sample['labels']
                break

        boxes = torch.stack(tuple(
            map(torch.tensor, zip(*boxes))
        )).permute(1, 0)
        boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]]  # yxyx: be warning

        target = dict(
            boxes=boxes,
            labels=torch.tensor(labels),
        )

        return image, target

    def load_image_and_boxes(self, index):
        image_name = self.image_names[index]
        image = read_img_cv2(self.images_dir / image_name)

        records = self.df[self.df['image_name'] == image_name].copy()
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
        return image, boxes, labels

    def __len__(self):
        return len(self.image_names)


class AdditionalImageDataset(ImageDataset):
    def __init__(self, root, df, transforms):
        self.images_dir = root / 'images'
        self.df = df
        self.image_names = list(df['image'].unique())
        self.transforms = transforms

    def load_image_and_boxes(self, index):
        image_name = self.image_names[index]
        image = read_img_cv2(self.images_dir / image_name)

        records = self.df[self.df['image'] == image_name].copy()
        boxes = records[['left', 'top', 'width', 'height']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        labels = np.ones(len(boxes))
        return image, boxes, labels

    def __len__(self):
        return len(self.image_names)

class ValidationImageDataset(Dataset):

    def __init__(self, video_path, df, transforms):
        self.video_path = Path(video_path)
        self.df = df[df['video'] == self.video_path.name]
        self.transforms = albu.Compose(list(transforms))

        self.video = read_video(video_path)

    def __getitem__(self, index):
        frame_num = index + 1

        image, boxes = self.load_image_and_boxes(index)
        image = self.transforms(image=image)['image']

        return image, boxes, frame_num

    def load_image_and_boxes(self, index):
        image = self.video[index]

        records = self.df[self.df['frame'] == (index + 1)].copy()
        records = records[
            (records['impact'] > 1) &
            (records['confidence'] > 1) &
            (records['visibility'] > 0)
            ]
        boxes = records[['left', 'top', 'width', 'height']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        return image, boxes

    def __len__(self):
        return len(self.video)
