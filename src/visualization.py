import cv2

from src.data.utils import read_video, write_video


def draw_hits_video(video_path, preds, scores, out_path=None):
    video = read_video(video_path)

    for pred, score in zip(preds, scores):
        frame = pred[0]
        box = pred[1:]

        cv2.rectangle(video[frame], (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

    if out_path is not None:
        write_video(video, out_path)

    return video