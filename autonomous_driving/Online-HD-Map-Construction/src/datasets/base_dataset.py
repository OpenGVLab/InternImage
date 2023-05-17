import numpy as np
import os
import os.path as osp
import mmcv
from .evaluation.vector_eval import VectorEvaluate

from mmdet3d.datasets.pipelines import Compose
from mmdet.datasets import DATASETS
from torch.utils.data import Dataset
import warnings

warnings.filterwarnings("ignore")

@DATASETS.register_module()
class BaseMapDataset(Dataset):
    """Map dataset base class.

    Args:
        ann_file (str): annotation file path
        cat2id (dict): category to class id
        roi_size (tuple): bev range
        eval_config (Config): evaluation config
        meta (dict): meta information
        pipeline (Config): data processing pipeline config,
        interval (int): annotation load interval
        work_dir (str): path to work dir
        test_mode (bool): whether in test mode
    """
    def __init__(self, 
                 ann_file,
                 root_path,
                 cat2id,
                 roi_size,
                 meta,
                 pipeline,
                 interval=1,
                 work_dir=None,
                 test_mode=False,
        ):
        super().__init__()
        self.ann_file = ann_file
        self.meta = meta
        self.root_path = root_path
        
        self.classes = list(cat2id.keys())
        self.num_classes = len(self.classes)
        self.cat2id = cat2id
        self.interval = interval

        self.load_annotations(self.ann_file)
        self.idx2token = {}
        for i, s in enumerate(self.samples):
            if 'timestamp' in s:
                self.idx2token[i] = s['timestamp']
            else:
                self.idx2token[i] = s['token']
        self.token2idx = {v: k for k, v in self.idx2token.items()}

        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        else:
            self.pipeline = None
        
        # dummy flags to fit with mmdet dataset
        self.flag = np.zeros(len(self), dtype=np.uint8)

        self.roi_size = roi_size
        
        self.work_dir = work_dir
        self.test_mode = test_mode

    def load_annotations(self, ann_file):
        raise NotImplementedError

    def get_sample(self, idx):
        raise NotImplementedError

    def format_results(self, results, denormalize=True, prefix=None):
        '''Format prediction result to submission format.
        
        Args:
            results (list[Tensor]): List of prediction results.
            denormalize (bool): whether to denormalize prediction from (0, 1) \
                to bev range. Default: True
            prefix (str): work dir prefix to save submission file.

        Returns:
            dict: Evaluation results
        '''

        meta = self.meta
        submissions = {
            'meta': meta,
            'results': {},
        }

        for pred in results:
            '''
            For each case, the result should be formatted as Dict{'vectors': [], 'scores': [], 'labels': []}
            'vectors': List of vector, each vector is a array([[x1, y1], [x2, y2] ...]),
                contain all vectors predicted in this sample.
            'scores: List of score(float), 
                contain scores of all instances in this sample.
            'labels': List of label(int), 
                contain labels of all instances in this sample.
            '''
            if pred is None: # empty prediction
                continue
            
            single_case = {'vectors': [], 'scores': [], 'labels': []}
            token = pred['token']
            roi_size = np.array(self.roi_size)
            origin = -np.array([self.roi_size[0]/2, self.roi_size[1]/2])

            for i in range(len(pred['scores'])):
                score = pred['scores'][i]
                label = pred['labels'][i]
                vector = pred['vectors'][i]

                # A line should have >=2 points
                if len(vector) < 2:
                    continue
                
                if denormalize:
                    eps = 2
                    vector = vector * (roi_size + eps) + origin

                single_case['vectors'].append(vector)
                single_case['scores'].append(score)
                single_case['labels'].append(label)
            
            submissions['results'][token] = single_case
        
        out_path = osp.join(prefix, 'submission_vector.json')
        print(f'\nsaving submissions results to {out_path}')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        mmcv.dump(submissions, out_path)
        return out_path

    def evaluate(self, results, logger=None, **kwargs):
        '''Evaluate prediction result based on `output_format` specified by dataset.

        Args:
            results (list[Tensor]): List of prediction results.
            logger (logger): logger to print evaluation results.

        Returns:
            dict: Evaluation results.
        '''

        output_format = self.meta['output_format']
        self.evaluator = VectorEvaluate(self.ann_file)

        print('len of the results', len(results))
        
        result_path = self.format_results(results, denormalize=True, prefix=self.work_dir)

        result_dict = self.evaluator.evaluate(result_path, logger=logger)
        return result_dict

    def __len__(self):
        """Return the length of data infos.

        Returns:
            int: Length of data infos.
        """
        return len(self.samples)
        
    def _rand_another(self, idx):
        """Randomly get another item.

        Returns:
            int: Another index of item.
        """
        return np.random.choice(self.__len__)

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        input_dict = self.get_sample(idx)
        data = self.pipeline(input_dict)
        return data

