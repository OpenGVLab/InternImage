import numpy as np
from .distance import chamfer_distance, frechet_distance
from typing import List, Tuple, Union
from numpy.typing import NDArray

def average_precision(recalls, precisions, mode='area'):
    """Calculate average precision. 

    Args:
        recalls (ndarray): shape (num_dets, )
        precisions (ndarray): shape (num_dets, )
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]

    Returns:
        float: calculated average precision
    """

    recalls = recalls[np.newaxis, :]
    precisions = precisions[np.newaxis, :]

    assert recalls.shape == precisions.shape and recalls.ndim == 2
    num_scales = recalls.shape[0]
    ap = 0.

    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        
        ind = np.where(mrec[0, 1:] != mrec[0, :-1])[0]
        ap = np.sum(
            (mrec[0, ind + 1] - mrec[0, ind]) * mpre[0, ind + 1])
    
    elif mode == '11points':
        for thr in np.arange(0, 1 + 1e-3, 0.1):
            precs = precisions[0, recalls[i, :] >= thr]
            prec = precs.max() if precs.size > 0 else 0
            ap += prec
        ap /= 11
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported')
    
    return ap

def instance_match(pred_lines: List[NDArray], 
                   scores: NDArray, 
                   gt_lines: List[NDArray], 
                   thresholds: Union[Tuple, List], 
                   metric: str='chamfer') -> List:
    """Compute whether detected lines are true positive or false positive.

    Args:
        pred_lines (List): Detected lines of a sample, each line has shape (INTERP_NUM, 2 or 3).
        scores (array): Confidence score of each line, of shape (M, ).
        gt_lines (List): GT lines of a sample, each line has shape (INTERP_NUM, 2 or 3).
        thresholds (list of tuple): List of thresholds.
        metric (str): Distance function for lines matching. Default: 'chamfer'.

    Returns:
        list_of_tp_fp (list): tp-fp matching result at all thresholds
    """

    if metric == 'chamfer':
        distance_fn = chamfer_distance

    elif metric == 'frechet':
        distance_fn = frechet_distance
    
    else:
        raise ValueError(f'unknown distance function {metric}')

    num_preds = len(pred_lines)
    num_gts = len(gt_lines)

    # tp and fp
    tp_fp_list = []
    tp = np.zeros((num_preds), dtype=np.float32)
    fp = np.zeros((num_preds), dtype=np.float32)

    # if there is no gt lines in this sample, then all pred lines are false positives
    if num_gts == 0:
        fp[...] = 1
        for thr in thresholds:
            tp_fp_list.append((tp.copy(), fp.copy()))
        return tp_fp_list
    
    if num_preds == 0:
        for thr in thresholds:
            tp_fp_list.append((tp.copy(), fp.copy()))
        return tp_fp_list

    # distance matrix: M x N
    matrix = np.zeros((num_preds, num_gts))

    for i in range(num_preds):
        for j in range(num_gts):
            matrix[i, j] = distance_fn(pred_lines[i], gt_lines[j])

    # for each det, the min distance with all gts
    matrix_min = matrix.min(axis=1)

    # for each det, which gt is the closest to it
    matrix_argmin = matrix.argmin(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-scores)

    # match under different thresholds
    for thr in thresholds:
        tp = np.zeros((num_preds), dtype=np.float32)
        fp = np.zeros((num_preds), dtype=np.float32)

        gt_covered = np.zeros(num_gts, dtype=bool)
        for i in sort_inds:
            if matrix_min[i] <= thr:
                matched_gt = matrix_argmin[i]
                if not gt_covered[matched_gt]:
                    gt_covered[matched_gt] = True
                    tp[i] = 1
                else:
                    fp[i] = 1
            else:
                fp[i] = 1
        
        tp_fp_list.append((tp, fp))

    return tp_fp_list