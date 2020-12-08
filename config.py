import math
from pathlib import Path

import albumentations as A
import torch
from albumentations.pytorch.transforms import ToTensorV2
from easydict import EasyDict


def get_padded_size(side, div=128):
    return math.ceil(side / div) * div


# img_size = (720, 1280)
img_size = (360, 640)
padded_img_size = (get_padded_size(img_size[0]), get_padded_size(img_size[1]))

pre_transforms = [
    A.Resize(height=img_size[0], width=img_size[1]),
]
augmentations = [
    A.Cutout(num_holes=200, max_h_size=img_size[0] // 72, max_w_size=img_size[0] // 72, p=0.3),
    A.OneOf([
        A.ChannelShuffle(p=1.0),
        A.ChannelDropout(p=1.0),
        A.ToGray(p=1.0),
    ], p=0.3),

    A.OneOf([
        A.RGBShift(
            r_shift_limit=20,
            g_shift_limit=20,
            b_shift_limit=20,
            p=1.0),
        A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=1.0)
    ], p=0.5),
    A.RandomBrightnessContrast(
        brightness_limit=0.3,
        contrast_limit=0.1, p=0.5
    ),

    A.GaussNoise(var_limit=(10, 50), p=0.5),
    A.OneOf([
        A.MotionBlur(blur_limit=img_size[0] // 36, p=1.0),
        A.Blur(blur_limit=img_size[0] // 102, p=1.0),
        A.MedianBlur(blur_limit=img_size[0] // 102, p=1.0)
    ], p=0.2),
]

post_transforms = [
    A.PadIfNeeded(
        min_height=padded_img_size[0],
        min_width=padded_img_size[1],
        border_mode=0,
        p=1.0
    ),
    A.Normalize(),
    ToTensorV2(p=1.0),
]
bbox_params = A.BboxParams(
    format='pascal_voc',
    min_area=0,
    min_visibility=0,
    label_fields=['labels']
)

train_pipeline = A.Compose(
    pre_transforms + augmentations + post_transforms,
    bbox_params=bbox_params,
)
valid_pipeline = A.Compose(
    pre_transforms + post_transforms,
    bbox_params=bbox_params,
)

data = EasyDict(dict(
    root=Path('/dataset/nfl'),
    train_only_accidents=False,
    frames_neighbors=(-12, -9, -6, -3, 0, 3, 6, 9, 12),
    train_pipeline=train_pipeline,
    valid_pipeline=valid_pipeline,
))

train = EasyDict(dict(
    num_workers=7,
    batch_size=7,
    valid_batch_size=7,
    lr_per_image=0.0001,
    n_epochs=30,

    step_scheduler=False,
    validation_scheduler=True,
    SchedulerClass=torch.optim.lr_scheduler.ReduceLROnPlateau,
    scheduler_params=dict(
        mode='max',
        factor=0.5,
        patience=1,
        verbose=False,
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0,
        min_lr=1e-8,
        eps=1e-08
    ),
    early_stopping_patience=5,

    verbose=True,
    verbose_step=1,
))
train.lr = train.lr_per_image * train.batch_size

model = EasyDict(dict(
    efficientdet_config='tf_efficientdet_d5',
    img_size=padded_img_size,

    pretrained_effdet='',
    pretrained_backbone_3d='',
    # if `start_from` is not empty then these weights overwrites pretrained weights.
    start_from='',

    freeze_backbone_2d=False,
    freeze_backbone_3d=False,
))
