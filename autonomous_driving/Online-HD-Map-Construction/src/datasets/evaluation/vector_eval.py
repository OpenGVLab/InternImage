from functools import partial
import numpy as np
from multiprocessing import Pool
from mmdet3d.datasets import build_dataset, build_dataloader
import mmcv
from .AP import instance_match, average_precision
import prettytable
from time import time
from functools import cached_property
from shapely.geometry import LineString
from numpy.typing import NDArray
from typing import Dict, List, Optional
from logging import Logger
from mmcv import Config
from copy import deepcopy

INTERP_NUM = 100 # number of points to interpolate during evaluation
SAMPLE_DIST = 0.3 # fixed sample distance
THRESHOLDS = [0.5, 1.0, 1.5] # AP thresholds
N_WORKERS = 16 # num workers to parallel

CAT2ID = {
    'ped_crossing': 0,
    'divider': 1,
    'boundary': 2,
}

class VectorEvaluate(object):
    """Evaluator for vectorized map.

    Args:
        dataset_cfg (Config): dataset cfg for gt
        n_workers (int): num workers to parallel
    """

    def __init__(self, ann_file, n_workers: int=N_WORKERS) -> None:
        ann = mmcv.load(ann_file)
        gts = {}
        for seg_id, seq in ann.items():
            for frame in seq:
                ann = {}
                for cat, vectors in frame['annotation'].items():
                    # only evaluate in 2-dimension
                    ann[cat] = [np.array(v)[:, :2] for v in vectors]
                    
                gts[frame['timestamp']] = ann
        
        self.gts = gts
        self.n_workers = n_workers
        self.cat2id = CAT2ID
        self.id2cat = {v: k for k, v in self.cat2id.items()}
    
    def interp_fixed_num(self, 
                         vector: NDArray, 
                         num_pts: int) -> NDArray:
        ''' Interpolate a polyline.
        
        Args:
            vector (array): line coordinates, shape (M, 2)
            num_pts (int): 
        
        Returns:
            sampled_points (array): interpolated coordinates
        '''
        line = LineString(vector)
        distances = np.linspace(0, line.length, num_pts)
        sampled_points = np.array([list(line.interpolate(distance).coords) 
            for distance in distances]).squeeze()
        
        return sampled_points
    
    def interp_fixed_dist(self, 
                          vector: NDArray,
                          sample_dist: float) -> NDArray:
        ''' Interpolate a line at fixed interval.
        
        Args:
            vector (LineString): vector
            sample_dist (float): sample interval
        
        Returns:
            points (array): interpolated points, shape (N, 2)
        '''
        line = LineString(vector)
        distances = list(np.arange(sample_dist, line.length, sample_dist))
        # make sure to sample at least two points when sample_dist > line.length
        distances = [0,] + distances + [line.length,] 
        
        sampled_points = np.array([list(line.interpolate(distance).coords)
                                for distance in distances]).squeeze()
        
        return sampled_points

    def _evaluate_single(self, 
                         pred_vectors: List, 
                         scores: List, 
                         groundtruth: List, 
                         thresholds: List, 
                         metric: str='metric') -> Dict[int, NDArray]:
        ''' Do single-frame matching for one class.
        
        Args:
            pred_vectors (List): List[vector(ndarray) (different length)], 
            scores (List): List[score(float)]
            groundtruth (List): List of vectors
            thresholds (List): List of thresholds
        
        Returns:
            tp_fp_score_by_thr (Dict): matching results at different thresholds
                e.g. {0.5: (M, 2), 1.0: (M, 2), 1.5: (M, 2)}
        '''

        pred_lines = []

        # interpolate predictions
        for vector in pred_vectors:
            vector = np.array(vector)
            # vector_interp = self.interp_fixed_num(vector, INTERP_NUM)
            vector_interp = self.interp_fixed_dist(vector, SAMPLE_DIST)
            pred_lines.append(vector_interp)

        # interpolate groundtruth
        gt_lines = []
        for vector in groundtruth:
            # vector_interp = self.interp_fixed_num(vector, INTERP_NUM)
            vector_interp = self.interp_fixed_dist(vector, SAMPLE_DIST)
            gt_lines.append(vector_interp)
        
        scores = np.array(scores)
        tp_fp_list = instance_match(pred_lines, scores, gt_lines, thresholds, metric) # (M, 2)
        tp_fp_score_by_thr = {}
        for i, thr in enumerate(thresholds):
            tp, fp = tp_fp_list[i]
            tp_fp_score = np.hstack([tp[:, None], fp[:, None], scores[:, None]])
            tp_fp_score_by_thr[thr] = tp_fp_score
        
        return tp_fp_score_by_thr # {0.5: (M, 2), 1.0: (M, 2), 1.5: (M, 2)}
        
    def evaluate(self, 
                 result_path: str, 
                 metric: str='chamfer', 
                 logger: Optional[Logger]=None) -> Dict[str, float]:
        ''' Do evaluation for a submission file and print evalution results to `logger` if specified.
        The submission will be aligned by tokens before evaluation. We use multi-worker to speed up.
        
        Args:
            result_path (str): path to submission file
            metric (str): distance metric. Default: 'chamfer'
            logger (Logger): logger to print evaluation result, Default: None
        
        Returns:
            new_result_dict (Dict): evaluation results. AP by categories.
        '''
        
        results = mmcv.load(result_path)
        results = results['results']
        
        # re-group samples and gt by label
        samples_by_cls = {label: [] for label in self.id2cat.keys()}
        num_gts = {label: 0 for label in self.id2cat.keys()}
        num_preds = {label: 0 for label in self.id2cat.keys()}

        # align by token
        for token, gt in self.gts.items():
            if token in results.keys():
                pred = results[token]
            else:
                pred = {'vectors': [], 'scores': [], 'labels': []}
            
            # for every sample
            vectors_by_cls = {label: [] for label in self.id2cat.keys()}
            scores_by_cls = {label: [] for label in self.id2cat.keys()}

            for i in range(len(pred['labels'])):
                # i-th pred line in sample
                label = pred['labels'][i]
                vector = pred['vectors'][i]
                score = pred['scores'][i]

                vectors_by_cls[label].append(vector)
                scores_by_cls[label].append(score)

            for label, cat in self.id2cat.items():
                new_sample = (vectors_by_cls[label], scores_by_cls[label], gt[cat])
                num_gts[label] += len(gt[cat])
                num_preds[label] += len(scores_by_cls[label])
                samples_by_cls[label].append(new_sample)

        result_dict = {}

        print(f'\nevaluating {len(self.id2cat)} categories...')
        start = time()
        if self.n_workers > 0:
            pool = Pool(self.n_workers)
        
        sum_mAP = 0
        pbar = mmcv.ProgressBar(len(self.id2cat))
        for label in self.id2cat.keys():
            samples = samples_by_cls[label] # List[(pred_lines, scores, gts)]
            result_dict[self.id2cat[label]] = {
                'num_gts': num_gts[label],
                'num_preds': num_preds[label]
            }
            sum_AP = 0

            fn = partial(self._evaluate_single, thresholds=THRESHOLDS, metric=metric)
            if self.n_workers > 0:
                tpfp_score_list = pool.starmap(fn, samples)
            else:
                tpfp_score_list = []
                for sample in samples:
                    tpfp_score_list.append(fn(*sample))
            
            for thr in THRESHOLDS:
                tp_fp_score = [i[thr] for i in tpfp_score_list]
                tp_fp_score = np.vstack(tp_fp_score) # (num_dets, 3)
                sort_inds = np.argsort(-tp_fp_score[:, -1])

                tp = tp_fp_score[sort_inds, 0] # (num_dets,)
                fp = tp_fp_score[sort_inds, 1] # (num_dets,)
                tp = np.cumsum(tp, axis=0)
                fp = np.cumsum(fp, axis=0)
                eps = np.finfo(np.float32).eps
                recalls = tp / np.maximum(num_gts[label], eps)
                precisions = tp / np.maximum((tp + fp), eps)

                AP = average_precision(recalls, precisions, 'area')
                sum_AP += AP
                result_dict[self.id2cat[label]].update({f'AP@{thr}': AP})

            pbar.update()
            
            AP = sum_AP / len(THRESHOLDS)
            sum_mAP += AP

            result_dict[self.id2cat[label]].update({f'AP': AP})
        
        if self.n_workers > 0:
            pool.close()
        
        mAP = sum_mAP / len(self.id2cat.keys())
        result_dict.update({'mAP': mAP})
        
        print(f"finished in {time() - start:.2f}s")

        # print results
        table = prettytable.PrettyTable(['category', 'num_preds', 'num_gts'] + 
                [f'AP@{thr}' for thr in THRESHOLDS] + ['AP'])
        for label in self.id2cat.keys():
            table.add_row([
                self.id2cat[label], 
                result_dict[self.id2cat[label]]['num_preds'],
                result_dict[self.id2cat[label]]['num_gts'],
                *[round(result_dict[self.id2cat[label]][f'AP@{thr}'], 4) for thr in THRESHOLDS],
                round(result_dict[self.id2cat[label]]['AP'], 4),
            ])
        
        from mmcv.utils import print_log
        print_log('\n'+str(table), logger=logger)
        print_log(f'mAP = {mAP:.4f}\n', logger=logger)

        new_result_dict = {}
        for name in self.cat2id:
            new_result_dict[name] = result_dict[name]['AP']

        return new_result_dict