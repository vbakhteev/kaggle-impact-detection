from functools import partial

import pandas as pd
from torch.utils.data import ConcatDataset, DataLoader

from .clip_dataset import ClipDataset, ValidationClipDataset
from .image_dataset import ImageDataset, AdditionalImageDataset, ValidationImageDataset
from .utils import collate_fn


def get_dataloaders(data_cfg, train_cfg):
    root = data_cfg.root

    df = pd.read_csv(root / 'updated_train_labels.csv')
    df['impact'] = df['impact'] + 1
    valid_split = pd.read_csv(root / 'validation_split.csv')
    df = df.merge(valid_split, left_on='gameKey', right_on='gameKey')
    df_train = df[df['train'] == 1]

    datasets = []
    for video_name in sorted(df_train['video'].unique()):
        dataset = ClipDataset(
            video_path=root / 'train' / video_name,
            neighbors=data_cfg.frames_neighbors,
            df=df_train[df_train['video'] == video_name],
            only_accidents=data_cfg.train_only_accidents,
            transforms=data_cfg.train_pipeline,
        )
        datasets += [dataset]

    train_dataset = ConcatDataset(datasets)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    valid_loaders_fn = partial(
        get_valid_dataloaders,
        data_cfg=data_cfg, train_cfg=train_cfg,
    )

    return train_loader, valid_loaders_fn


def get_valid_dataloaders(data_cfg, train_cfg):
    root = data_cfg.root
    df = pd.read_csv(root / 'train_labels.csv')
    df['impact'] = df['impact'] + 1
    valid_split = pd.read_csv(root / 'validation_split.csv')
    df = df.merge(valid_split, left_on='gameKey', right_on='gameKey')
    df = df[df['train'] == 0]


    for video_name in df['video'].unique():
        df_video = df[df['video'] == video_name]
        dataset = ValidationClipDataset(
            video_path=root / 'train' / video_name,
            neighbors=data_cfg.frames_neighbors,
            df=df_video,
            transforms=data_cfg.valid_pipeline,
        )
        loader = DataLoader(
            dataset,
            batch_size=train_cfg.valid_batch_size,
            num_workers=train_cfg.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
        yield loader


def get_image_dataloaders(data_cfg, train_cfg):
    root = data_cfg.root

    df = pd.read_csv(root / 'updated_train_labels.csv')
    df['impact'] = df['impact'] + 1
    valid_split = pd.read_csv(root / 'validation_split.csv')
    df = df.merge(valid_split, left_on='gameKey', right_on='gameKey')
    df_train = df[df['train'] == 1]
    train_dataset1 = ImageDataset(
        root=root,
        df=df_train,
        only_accidents=data_cfg.train_only_accidents,
        transforms=data_cfg.train_pipeline,
    )

    df_images = pd.read_csv(root / 'image_labels.csv')
    train_dataset2 = AdditionalImageDataset(
        root=root,
        df=df_images,
        transforms=data_cfg.train_pipeline,
    )

    train_dataset = ConcatDataset([train_dataset1, train_dataset2])
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    valid_loaders_fn = partial(
        get_valid_image_dataloaders,
        data_cfg=data_cfg, train_cfg=train_cfg,
    )

    return train_loader, valid_loaders_fn


def get_valid_image_dataloaders(data_cfg, train_cfg):
    root = data_cfg.root
    df = pd.read_csv(root / 'train_labels.csv')
    df['impact'] = df['impact'] + 1
    valid_split = pd.read_csv(root / 'validation_split.csv')
    df = df.merge(valid_split, left_on='gameKey', right_on='gameKey')
    df = df[df['train'] == 0]

    for video_name in df['video'].unique():
        df_video = df[df['video'] == video_name]
        dataset = ValidationImageDataset(
            video_path=root / 'train' / video_name,
            df=df_video,
            transforms=data_cfg.valid_pipeline,
        )
        loader = DataLoader(
            dataset,
            batch_size=train_cfg.valid_batch_size,
            num_workers=train_cfg.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
        yield loader