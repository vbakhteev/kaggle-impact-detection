import numpy as np
from scipy.optimize import linear_sum_assignment


def comp_metric(preds, gt):
    '''
    [
     [[frame_index, xmin, ymin, xmax, ymax] for _ in range(n_bboxes)]
     for n in range(n_videos)
    ]
    '''

    ftp, ffp, ffn = [], [], []
    for pred_boxes, gt_boxes in zip(preds, gt):
        # gt_boxes: [[frame_index, xmin, ymin, xmax, ymax] for _ in range(n_bboxes)]
        # pred_boxes: same
        tp, fp, fn = precision_calc(gt_boxes, pred_boxes)
        ftp.append(tp)
        ffp.append(fp)
        ffn.append(fn)

    tp = np.sum(ftp)
    fp = np.sum(ffp)
    fn = np.sum(ffn)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)

    tp = np.array(ftp)
    fp = np.array(ffp)
    fn = np.array(ffn)
    precision_per_video = tp / (tp + fp + 1e-6)
    recall_per_video = tp / (tp + fn + 1e-6)
    f1_per_video = 2 * (precision_per_video * recall_per_video) / (precision_per_video + recall_per_video + 1e-6)

    return precision, recall, f1_score, f1_per_video


def iou(bbox1, bbox2):
    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union


def precision_calc(gt_boxes, pred_boxes):  # frame_index, xmin, ymin, xmax, ymax
    cost_matix = np.ones((len(gt_boxes), len(pred_boxes)))
    for i, box1 in enumerate(gt_boxes):
        for j, box2 in enumerate(pred_boxes):
            dist = abs(box1[0] - box2[0])
            if dist > 4:
                continue
            iou_score = iou(box1[1:], box2[1:])

            if iou_score < 0.35:
                continue
            else:
                cost_matix[i, j] = 0

    row_ind, col_ind = linear_sum_assignment(cost_matix)
    fn = len(gt_boxes) - row_ind.shape[0]
    fp = len(pred_boxes) - col_ind.shape[0]
    tp = 0
    for i, j in zip(row_ind, col_ind):
        if cost_matix[i, j] == 0:
            tp += 1
        else:
            fp += 1
            fn += 1
    return tp, fp, fn
