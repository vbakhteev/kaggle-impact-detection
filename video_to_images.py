import os
import shutil
from functools import partial
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import data as data_cfg


def mk_images(video_name, video_labels, video_dir, out_dir, only_with_impact=True):
    video_path = video_dir / video_name
    video_name = os.path.basename(video_path)
    vidcap = cv2.VideoCapture(str(video_path))
    frame = 0

    if only_with_impact:
        boxes_all = video_labels.query("video == @video_name")
        print(video_path, boxes_all[boxes_all.impact > 0].shape[0])
    else:
        print(video_path)

    while True:
        it_worked, img = vidcap.read()
        if not it_worked:
            break
        frame += 1

        if only_with_impact:
            boxes = video_labels.query("video == @video_name and frame == @frame")
            boxes_with_impact = boxes[boxes.impact > 0]
            if boxes_with_impact.shape[0] == 0:
                continue

        image_path = out_dir / video_name.replace('.mp4', f'_{frame}.png')
        _ = cv2.imwrite(str(image_path), img)


def main():
    only_with_impact = False

    root = data_cfg['root']

    video_labels = pd.read_csv(root / 'train_labels.csv').fillna(0)
    video_labels_with_impact = video_labels[video_labels['impact'] > 0]
    for row in tqdm(video_labels_with_impact[['video', 'frame', 'label']].values):
        frames = np.array([-4, -3, -2, -1, 1, 2, 3, 4]) + row[1]
        video_labels.loc[(video_labels['video'] == row[0])
                         & (video_labels['frame'].isin(frames))
                         & (video_labels['label'] == row[2]), 'impact'] = 1
    video_labels['image_name'] = video_labels['video'].str.replace('.mp4', '') + '_' + video_labels['frame'].astype(
        str) + '.png'

    uniq_video = video_labels.video.unique()
    video_dir = root / 'train'
    out_dir = root / 'train_images'
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)

    if only_with_impact:
        video_labels = video_labels[video_labels.groupby('image_name')['impact'].transform("sum") > 0].reset_index(
            drop=True)

    mk_images_fn = partial(
        mk_images,
        video_labels=video_labels,
        video_dir=video_dir,
        out_dir=out_dir,
        only_with_impact=only_with_impact,
    )
    with Pool(16) as p:
        p.map(mk_images_fn, uniq_video)

    video_labels.to_csv(root / 'updated_train_labels.csv', index=False)


if __name__ == '__main__':
    main()