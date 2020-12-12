import random

import albumentations as albu
import torch


@torch.no_grad()
def mixup_video(images1, images2, boxes1, boxes2, labels1, labels2):
    images = (images1 + images2) / 2
    boxes = torch.cat([boxes1, boxes2])
    labels = torch.cat([labels1, labels2])
    return images, boxes, labels


@torch.no_grad()
def cutmix_video(images1, images2, boxes1, boxes2, labels1, labels2):
    # TODO implement cutmix
    return images1, boxes1, labels1


class VideoTranform:
    def __init__(self, augmentation, p=0.5):
        self.augmentation = augmentation
        self.p = p
        self.bbox_params = albu.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )

    def __call__(self, images, bboxes, labels):
        augmentation = albu.Compose([self.augmentation], bbox_params=self.bbox_params, p=1)

        if random.random() < self.p:
            for i in range(len(images)):
                sample = augmentation(
                    image=images[i],
                    bboxes=bboxes,
                    labels=labels,
                )
                images[i] = sample['image']

            bboxes = sample['bboxes']
            labels = sample['labels']

        return images, bboxes, labels


class HorizontalFlip(VideoTranform):
    def __init__(self):
        aug = albu.HorizontalFlip(p=1)
        super().__init__(aug, p=0.5)


class ShiftScale(VideoTranform):
    def __init__(self, shift_limit=(-0.1, 0.1), scale_limit=(-0.5, 0.5), p=0.5):
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        aug = None
        super().__init__(aug, p)

    def __call__(self, images, bboxes, labels):
        shift_limit = random.uniform(*self.shift_limit)
        scale_limit = random.uniform(*self.scale_limit)
        self.augmentation = albu.ShiftScaleRotate(
            shift_limit=(shift_limit, shift_limit),
            scale_limit=(scale_limit, scale_limit),
            rotate_limit=0,
            border_mode=0,
            p=1.0,
        )

        images, bboxes, labels = super().__call__(images, bboxes, labels)
        return images, bboxes, labels
