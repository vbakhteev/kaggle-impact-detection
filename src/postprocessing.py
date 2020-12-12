import numpy as np
from ensemble_boxes import (nms, soft_nms, non_maximum_weighted,
                            weighted_boxes_fusion, weighted_boxes_fusion_3d)

from src.metric import iou


def postprocessing_video(
        preds, scores, global_thr,
        nms_iou_thr=0.35,
        same_helmet_iou_thr=0.35,
        look_future_n_frames=14,
        track_max_len=10,
):
    """
    params:
        preds: (N, 5)
        scores: (N,)
        global_thr - threshold of scores before all postprocessing.
        nms_iou_thr - IoU value for boxes to be a match.
        same_helmet_iou_thr - during tracks generation, next box occurrence is assigned to track if
            IoU between last box and new box is bigger than this threshold.
        look_future_n_frames - if box not found for `look_future_n_frames` since last box in track,
            then terminate track.
        track_max_len - terminate track if it's len exceed this value.
    """
    preds, scores = threshold_preds(preds, scores, threshold=global_thr)
    preds, scores = filter_by_frame(preds, scores, iou_thr=nms_iou_thr)

    tracks = generate_tracks(
        preds, scores,
        same_helmet_iou_thr=same_helmet_iou_thr,
        look_future_n_frames=look_future_n_frames,
        track_max_len=track_max_len,
    )
    pred_ids = [track[len(track) // 2] for track in tracks]
    preds = preds[pred_ids]
    scores = scores[pred_ids]

    return preds, scores

def threshold_preds(preds, scores, threshold):
    cond = scores > threshold
    return preds[cond], scores[cond]

def filter_by_frame(preds, scores, iou_thr=0.25):
    """Apply nms for each frame independently
    params:
        preds: (N, 5)
        scores: (N,)
    """
    if preds.shape[0] == 0:
        return preds, scores

    frames = preds[:, 0]
    boxes = preds[:, 1:].astype(float)

    boxes[:, 0] = boxes[:, 0] / 1280
    boxes[:, 2] = boxes[:, 2] / 1280
    boxes[:, 1] = boxes[:, 1] / 720
    boxes[:, 3] = boxes[:, 3] / 720

    result_preds, result_scores = [], []
    for frame in sorted(list(set(frames))):
        cond = frames == frame

        boxes_frame = boxes[cond]
        scores_frame = scores[cond]

        boxes_nms, scores_nms, _ = nms(
            [boxes_frame],
            [scores_frame],
            [np.ones_like(scores_frame)],
            iou_thr=iou_thr,
        )

        boxes_nms[:, 0] = boxes_nms[:, 0] * 1280
        boxes_nms[:, 2] = boxes_nms[:, 2] * 1280
        boxes_nms[:, 1] = boxes_nms[:, 1] * 720
        boxes_nms[:, 3] = boxes_nms[:, 3] * 720
        boxes_nms = boxes_nms.astype(int)

        preds = np.concatenate([
            np.full((len(boxes_nms), 1), frame),
            boxes_nms
        ], axis=1)

        result_preds += [preds]
        result_scores += [scores_nms]

    if len(scores) == 0:
        return np.zeros((0, 5), dtype=int), np.zeros(0)

    result_preds = np.concatenate(result_preds)
    result_scores = np.concatenate(result_scores)
    return result_preds, result_scores


def generate_tracks(preds, scores, same_helmet_iou_thr=0.5, look_future_n_frames=8, track_max_len=8):
    N = preds.shape[0]
    tracks = []
    processed = set()

    def helper(i, track: list):
        processed.add(i)
        track.append(i)

        curr_pred = preds[i]
        curr_frame = curr_pred[0]
        curr_box = curr_pred[1:]

        is_track_short: bool = len(track) <= track_max_len
        while (i + 1 < N) and (preds[i + 1, 0] - curr_frame < look_future_n_frames):
            next_box_hit = iou(curr_box, preds[i + 1, 1:]) > same_helmet_iou_thr

            if next_box_hit and is_track_short and (i + 1 not in processed):
                helper(i + 1, track)
                return

            i += 1

    for i in range(N):
        if i in processed:
            continue

        track = []
        helper(i, track)
        tracks += [track]

    return tracks
