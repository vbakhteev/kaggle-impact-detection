import numpy as np
from ensemble_boxes import (nms, soft_nms, non_maximum_weighted,
                            weighted_boxes_fusion, weighted_boxes_fusion_3d)


def threshold_preds(preds, scores, threshold):
    cond = scores > threshold
    return preds[cond], scores[cond]

def filter_by_frame(preds, scores, iou_thr=0.25):
    """Apply nms for each frame independently
    params:
        preds: (N, 5)
        scores: (N,)
    """
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

    result_preds = np.concatenate(result_preds)
    result_scores = np.concatenate(result_scores)
    return result_preds, result_scores
