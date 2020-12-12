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
def cutmix_video(samples: list):
    """
    samples: [(images, boxes, labels)]. Len=2
    """
    # TODO implement cutmix
    images1, boxes1, labels1 = samples[0]
    return images1, boxes1, labels1


@torch.no_grad()
def cutmix_image(samples):
    """
    samples: [(image, boxes, labels)]. Len=4
    """
    h, w = samples[0][0].shape[1:]
    xs = w // 2
    ys = h // 2

    # center x, y
    xc = int(random.uniform(w * 0.25, w * 0.75))
    yc = int(random.uniform(h * 0.25, h * 0.75))

    result_image = torch.full((3, h, w), 0, dtype=torch.float32)
    result_boxes = []
    result_labels = []

    for i, sample in enumerate(samples):
        image, boxes, labels = sample
        boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]]  # xyxy

        if i == 0:
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, xs * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(ys * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, xs * 2), min(ys * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
        result_image[:, y1a:y2a, x1a:x2a] = image[:, y1b:y2b, x1b:x2b]
        padw = x1a - x1b
        padh = y1a - y1b

        boxes[:, 0] += padw
        boxes[:, 1] += padh
        boxes[:, 2] += padw
        boxes[:, 3] += padh

        result_boxes.append(boxes)
        result_labels.append(labels)

    result_boxes = torch.cat(result_boxes, 0)
    result_labels = torch.cat(result_labels)
    torch.clamp(result_boxes[:, 0], 0, 2 * xs, out=result_boxes[:, 0])
    torch.clamp(result_boxes[:, 2], 0, 2 * xs, out=result_boxes[:, 2])
    torch.clamp(result_boxes[:, 1], 0, 2 * ys, out=result_boxes[:, 1])
    torch.clamp(result_boxes[:, 3], 0, 2 * ys, out=result_boxes[:, 3])
    result_boxes = result_boxes.int()

    boxes_to_use = (result_boxes[:, 2] - result_boxes[:, 0]) * (result_boxes[:, 3] - result_boxes[:, 1]) > 0
    result_boxes = result_boxes[boxes_to_use]
    result_labels = result_labels[boxes_to_use]

    result_boxes[:, [0, 1, 2, 3]] = result_boxes[:, [1, 0, 3, 2]]  # yxyx
    return result_image, result_boxes, result_labels


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
