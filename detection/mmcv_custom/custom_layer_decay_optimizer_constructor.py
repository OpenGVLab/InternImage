# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
"""
Mostly copy-paste from BEiT library:
https://github.com/microsoft/unilm/blob/master/beit/semantic_segmentation/mmcv_custom/layer_decay_optimizer_constructor.py
"""

import json

from mmcv.runner import OPTIMIZER_BUILDERS, DefaultOptimizerConstructor
from mmcv.runner import get_dist_info
from mmdet.utils import get_root_logger


def get_num_layer_for_swin(var_name, num_max_layer, depths):
    if var_name.startswith("backbone.patch_embed"):
        return 0
    elif "level_embeds" in var_name:
        return 0
    elif var_name.startswith("backbone.layers") or var_name.startswith(
            "backbone.levels"):
        if var_name.split('.')[3] not in ['downsample', 'norm']:
            stage_id = int(var_name.split('.')[2])
            layer_id = int(var_name.split('.')[4])
            # layers for Swin-Large: [2, 2, 18, 2]
            if stage_id == 0:
                return layer_id + 1
            elif stage_id == 1:
                return layer_id + 1 + depths[0]
            elif stage_id == 2:
                return layer_id + 1 + depths[0] + depths[1]
            else:
                return layer_id + 1 + depths[0] + depths[1] + depths[2]
        else:
            stage_id = int(var_name.split('.')[2])
            if stage_id == 0:
                return 1 + depths[0]
            elif stage_id == 1:
                return 1 + depths[0] + depths[1]
            elif stage_id == 2:
                return 1 + depths[0] + depths[1] + depths[2]
            else:
                return 1 + depths[0] + depths[1] + depths[2]
    else:
        return num_max_layer - 1


@OPTIMIZER_BUILDERS.register_module()
class CustomLayerDecayOptimizerConstructor(DefaultOptimizerConstructor):

    def add_params(self, params, module, prefix='', is_dcn_module=None):
        """Add all parameters of module to the params list.
        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.
        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
            prefix (str): The prefix of the module
            is_dcn_module (int|float|None): If the current module is a
                submodule of DCN, `is_dcn_module` will be passed to
                control conv_offset layer's learning rate. Defaults to None.
        """
        parameter_groups = {}
        logger = get_root_logger()
        logger.info(self.paramwise_cfg)
        backbone_small_lr = self.paramwise_cfg.get('backbone_small_lr', False)
        dino_head = self.paramwise_cfg.get('dino_head', False)
        num_layers = self.paramwise_cfg.get('num_layers') + 2
        layer_decay_rate = self.paramwise_cfg.get('layer_decay_rate')
        depths = self.paramwise_cfg.get('depths')
        offset_lr_scale = self.paramwise_cfg.get('offset_lr_scale', 1.0)

        logger.info("Build CustomLayerDecayOptimizerConstructor %f - %d" %
                    (layer_decay_rate, num_layers))
        weight_decay = self.base_wd

        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or \
                    "relative_position" in name or \
                    "norm" in name or\
                    "sampling_offsets" in name:
                group_name = "no_decay"
                this_weight_decay = 0.
            else:
                group_name = "decay"
                this_weight_decay = weight_decay

            layer_id = get_num_layer_for_swin(name, num_layers, depths)
            if layer_id == num_layers - 1 and dino_head and \
                    ("sampling_offsets" in name or "reference_points" in name):
                group_name = "layer_%d_%s_0.1x" % (layer_id, group_name)
            elif "sampling_offsets" in name or "reference_points" in name:
                group_name = "layer_%d_%s_offset_lr_scale" % (layer_id,
                                                              group_name)
            else:
                group_name = "layer_%d_%s" % (layer_id, group_name)

            if group_name not in parameter_groups:
                scale = layer_decay_rate ** (num_layers - layer_id - 1)
                if scale < 1 and backbone_small_lr == True:
                    scale = scale * 0.1
                if "0.1x" in group_name:
                    scale = scale * 0.1
                if "offset_lr_scale" in group_name:
                    scale = scale * offset_lr_scale

                parameter_groups[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "param_names": [],
                    "lr_scale": scale,
                    "group_name": group_name,
                    "lr": scale * self.base_lr,
                }

            parameter_groups[group_name]["params"].append(param)
            parameter_groups[group_name]["param_names"].append(name)
        rank, _ = get_dist_info()
        if rank == 0:
            to_display = {}
            for key in parameter_groups:
                to_display[key] = {
                    "param_names": parameter_groups[key]["param_names"],
                    "lr_scale": parameter_groups[key]["lr_scale"],
                    "lr": parameter_groups[key]["lr"],
                    "weight_decay": parameter_groups[key]["weight_decay"],
                }
            logger.info("Param groups = %s" % json.dumps(to_display, indent=2))

        # state_dict = module.state_dict()
        # for group_name in parameter_groups:
        #     group = parameter_groups[group_name]
        #     for name in group["param_names"]:
        #         group["params"].append(state_dict[name])

        params.extend(parameter_groups.values())