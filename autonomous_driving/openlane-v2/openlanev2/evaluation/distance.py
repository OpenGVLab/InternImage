# ==============================================================================
# Binaries and/or source for the following packages or projects 
# are presented under one or more of the following open source licenses:
# distance.py    The OpenLane-V2 Dataset Authors    Apache License, Version 2.0
#
# Contact wanghuijie@pjlab.org.cn if you have any issue.
#
# Copyright (c) 2023 The OpenLane-v2 Dataset Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
from scipy.spatial.distance import cdist, euclidean
from similaritymeasures import frechet_dist


def pairwise(xs: list, ys: list, distance_function: callable, mask: np.ndarray = None, relax: bool = False) -> np.ndarray:
    r"""
    Calculate pairwise distance.

    Parameters
    ----------
    xs : list
        List of data in shape (X, ).
    ys : list
        List of data in shape (Y, ).
    distance_function : callable
        Function that computes distance between two instance.
    mask : np.ndarray
        Boolean mask in shape (X, Y).
    relax : bool
        Relax the result based on distance to ego vehicle.

    Returns
    -------
    np.ndarray
        Float in shape (X, Y),
        where array[i][j] denotes distance between instance xs[i] and ys[j].

    """
    result = np.ones((len(xs), len(ys)), dtype=np.float64) * 1024
    for i, x in enumerate(xs):
        ego_distance = min([euclidean(p, np.zeros_like(p)) for p in x])
        relaxation_factor = max(0.5, 1 - 5e-3 * ego_distance) if relax else 1.0
        for j, y in enumerate(ys):
            if mask is None or mask[i][j]:
                result[i][j] = distance_function(x, y) * relaxation_factor
    return result

def chamfer_distance(gt: np.ndarray, pred: np.ndarray) -> float:
    r"""
    Calculate Chamfer distance.

    Parameters
    ----------
    gt : np.ndarray
        Curve of (G, N) shape,
        where G is the number of data points,
        and N is the number of dimmensions.
    pred : np.ndarray
        Curve of (P, N) shape,
        where P is the number of points,
        and N is the number of dimmensions.

    Returns
    -------
    float
        Chamfer distance

    Notes
    -----
    Adapted from https://github.com/Mrmoore98/VectorMapNet_code/blob/810ae463377f8e724c90a732361a675bcd7cf53b/plugin/datasets/evaluation/precision_recall/tgfg.py#L139.

    """
    assert gt.ndim == pred.ndim == 2 and gt.shape[1] == pred.shape[1]

    dist_mat = cdist(pred, gt)

    dist_pred = dist_mat.min(-1).mean()
    dist_gt = dist_mat.min(0).mean()

    return (dist_pred + dist_gt) / 2


def frechet_distance(gt: np.ndarray, pred: np.ndarray) -> float:
    r"""
    Calculate discrete Frechet distance.

    Parameters
    ----------
    gt : np.ndarray
        Curve of (G, N) shape,
        where G is the number of data points,
        and N is the number of dimmensions.
    pred : np.ndarray
        Curve of (P, N) shape,
        where P is the number of points,
        and N is the number of dimmensions.

    Returns
    -------
    float
        discrete Frechet distance

    """
    assert gt.ndim == pred.ndim == 2 and gt.shape[1] == pred.shape[1]

    return frechet_dist(pred, gt, p=2)

def iou_distance(gt: np.ndarray, pred: np.ndarray) -> float:
    r"""
    Calculate IoU distance,
    which is 1 - IoU.

    Parameters
    ----------
    gt : np.ndarray
        Bounding box in form [[x1, y1], [x2, y2]].
    pred : np.ndarray
        Bounding box in form [[x1, y1], [x2, y2]].

    Returns
    -------
    float
        IoU distance

    """
    assert pred.shape == gt.shape == (2, 2)

    bxmin = max(pred[0][0], gt[0][0])
    bymin = max(pred[0][1], gt[0][1])
    bxmax = min(pred[1][0], gt[1][0])
    bymax = min(pred[1][1], gt[1][1])

    inter = max((bxmax - bxmin), 0) * max((bymax - bymin), 0)
    union = (pred[1][0] - pred[0][0]) * (pred[1][1] - pred[0][1]) + (gt[1][0] - gt[0][0]) * (gt[1][1] - gt[0][1]) - inter

    return 1 - inter / union
