# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

# -*- coding: utf-8 -*-
from .custom_layer_decay_optimizer_constructor import \
    CustomLayerDecayOptimizerConstructor

__all__ = ['CustomLayerDecayOptimizerConstructor']

if torch.__version__.startswith('1.11'):

    from mmcv.runner.hooks import HOOKS, Hook
    from mmcv.runner.optimizer.builder import OPTIMIZERS
    from mmdet.utils.util_distribution import ddp_factory  # noqa: F401,F403
    from torch.distributed.optim import ZeroRedundancyOptimizer

    class ZeroAdamW(ZeroRedundancyOptimizer):
        def __init__(self, params, optimizer_class=torch.optim.AdamW, **kwargs):
            super().__init__(params[0]['params'],
                             optimizer_class=optimizer_class,
                             parameters_as_bucket_view=True,
                             **kwargs)
            for i in range(1, len(params)):
                self.add_param_group(params[i])

    OPTIMIZERS.register_module()(ZeroAdamW)


    @HOOKS.register_module()
    class ZeroHook(Hook):
        def __init__(self, interval):
            self.interval = interval

        def after_epoch(self, runner):
            runner.optimizer.consolidate_state_dict(to=0)

        def after_train_iter(self, runner):
            if self.every_n_iters(runner, self.interval):
                runner.optimizer.consolidate_state_dict(to=0)
