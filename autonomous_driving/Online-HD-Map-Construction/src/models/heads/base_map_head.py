from abc import ABCMeta, abstractmethod

import torch.nn as nn
from mmcv.runner import auto_fp16
from mmcv.utils import print_log

from mmdet.utils import get_root_logger


class BaseMapHead(nn.Module, metaclass=ABCMeta):
    """Base class for mappers."""

    def __init__(self):
        super(BaseMapHead, self).__init__()
        self.fp16_enabled = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if pretrained is not None:
            logger = get_root_logger()
            print_log(f'load model from: {pretrained}', logger=logger)

    @auto_fp16(apply_to=('img', ))
    def forward(self, *args, **kwargs):
        pass
        
    @abstractmethod
    def loss(self, pred, gt):
        '''
        Compute loss
        Output:
            dict(
                loss: torch.Tensor
                log_vars: dict(
                    str: float,
                )
                num_samples: int
            )
        '''
        return
        
    @abstractmethod
    def post_process(self, pred):
        '''
        convert model predictions to vectorized outputs
        the output format should be consistent with the evaluation function
        '''
        return
