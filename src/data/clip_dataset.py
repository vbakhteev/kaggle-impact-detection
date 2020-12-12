import random
from pathlib import Path

import albumentations as albu
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import get_video_len, read_img_cv2, read_video
from .transforms import HorizontalFlip, ShiftScale, cutmix_video, mixup_video

class ClipDataset(Dataset):
    def __init__(self, video_path, neighbors: tuple, df, transforms=None, only_accidents=False, cutmix_mixup=False):
        self.video_path = Path(video_path)
        self.images_dir = self.video_path.parent.parent / 'train_images'
        self.df = df[df['video'] == self.video_path.name]
        self.transforms = transforms        # Color transforms
        self.only_accidents = only_accidents
        self.cutmix_mixup = cutmix_mixup

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
        images1, boxes1, labels1 = self.get_sample(index)

        if self.cutmix_mixup:
            random_index = random.randint(0, self.__len__() - 1)
            images2, boxes2, labels2 = self.get_sample(random_index)

            if random.random() < 0.5:
                images, boxes, labels = mixup_video(
                    images1, images2, boxes1, boxes2, labels1, labels2
                )
            else:
                images, boxes, labels = cutmix_video(
                    images1, images2, boxes1, boxes2, labels1, labels2
                )

        else:
            images = images1
            boxes = boxes1
            labels = labels1

        return images, {'boxes': boxes, 'labels': labels}

    def get_sample(self, index):
        index += abs(self.min)
        images, boxes, labels = self.load_images_and_boxes(index)

        images, boxes, labels = self.shift_scale(*self.flip(images, boxes, labels))
        # Color augmentation for each frame independently
        for i in range(len(images)):
            sample = self.transforms(
                image=images[i],
                bboxes=boxes,
                labels=labels,
            )
            images[i] = sample['image']

        boxes = sample['bboxes']
        boxes = torch.stack(tuple(
            map(torch.tensor, zip(*boxes))
        )).permute(1, 0)
        boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]]  # yxyx: be warning

        return torch.stack(images, dim=1), boxes, torch.tensor(labels)


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

        self.video = read_video(video_path)


    def __getitem__(self, index):
        index += abs(self.min)
        frame_num = index + 1

        images, boxes = self.load_images_and_boxes(index)
        for i in range(len(images)):
            images[i] = self.transforms(image=images[i])['image']

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

