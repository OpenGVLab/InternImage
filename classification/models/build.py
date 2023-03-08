# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from .intern_image import InternImage


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'intern_image':
        model = InternImage(
            core_op=config.MODEL.INTERN_IMAGE.CORE_OP,
            num_classes=config.MODEL.NUM_CLASSES,
            channels=config.MODEL.INTERN_IMAGE.CHANNELS,
            depths=config.MODEL.INTERN_IMAGE.DEPTHS,
            groups=config.MODEL.INTERN_IMAGE.GROUPS,
            layer_scale=config.MODEL.INTERN_IMAGE.LAYER_SCALE,
            offset_scale=config.MODEL.INTERN_IMAGE.OFFSET_SCALE,
            post_norm=config.MODEL.INTERN_IMAGE.POST_NORM,
            mlp_ratio=config.MODEL.INTERN_IMAGE.MLP_RATIO,
            with_cp=config.TRAIN.USE_CHECKPOINT,
        )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
