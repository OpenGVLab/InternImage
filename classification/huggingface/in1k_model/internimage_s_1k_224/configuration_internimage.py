# --------------------------------------------------------
# InternImage
# Copyright (c) 2025 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from transformers import PretrainedConfig


class InternImageConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~InternImageModel`].
    It is used to instantiate an internimage model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    the internimage [OpenGVLab/internimage](https://huggingface.co/OpenGVLab/internimage) architecture.

    Configuration objects inherit from  [`PretrainedConfig`] and can be used
    to control the model outputs. Read the documentation from  [`PretrainedConfig`]
    for more information.

    Args:
        core_op (`str`, *optional*, defaults to `"DCNv3"`):
            Core operation used in the InternImageModel.
        depths (`tuple`, *optional*, defaults to `(4, 4, 18, 4)`):
            Tuple specifying the depth of layers in the InternImageModel.
        groups (`tuple`, *optional*, defaults to `(4, 8, 16, 32)`):
            Tuple specifying the group of layers in the InternImageModel.
        channels (`int`, *optional*, defaults to `64`):
            Number of channels in the InternImageModel.
        dw_kernel_size (`int`, *optional*, defaults to `None`):
            Kernel size for depthwise convolutions.
        layer_scale (`float`, *optional*, defaults to `None`):
            Scale of the layers in the model.
        offset_scale (`float`, *optional*, defaults to `1.0`):
            Offset scale in the model.
        mlp_ratio (`float`, *optional*, defaults to `4.0`):
            Ratio of mlp layers in the InternImageModel.
        post_norm (`bool`, *optional*, defaults to `False`):
            Whether to use post normalization in the model.
        level2_post_norm (`bool`, *optional*, defaults to `False`):
            Whether to use level 2 post normalization.
        level2_post_norm_block_ids (`list`, *optional*, defaults to `None`):
            Specific block IDs for level 2 post normalization.
        center_feature_scale (`bool`, *optional*, defaults to `False`):
            Whether to apply center feature scaling.
        use_clip_projector (`bool`, *optional*, defaults to `False`):
            Whether to use CLIP projector.
        remove_center (`bool`, *optional*, defaults to `False`):
            Whether to remove center pixels in some operations.
        num_classes (`int`, *optional*, defaults to `1000`):
            Number of classes for the model output.
        drop_rate (`float`, *optional*, defaults to `0.0`):
            Dropout rate in the model.
        drop_path_rate (`float`, *optional*, defaults to `0.0`):
            Dropout path rate in the model.
        drop_path_type (`str`, *optional*, defaults to `"linear"`):
            Type of dropout path used in the model.
        act_layer (`str`, *optional*, defaults to `"GELU"`):
            Activation function used in the model.
        norm_layer (`str`, *optional*, defaults to `"LN"`):
            Normalization layer used in the model.
        cls_scale (`float`, *optional*, defaults to `1.5`):
            Scale of the classification layer in the model.
        with_cp (`bool`, *optional*, defaults to `False`):
            Whether to use checkpointing in the model.
    """
    model_type = 'internimage'

    def __init__(
            self,
            core_op='DCNv3',
            depths=(4, 4, 18, 4),
            groups=(4, 8, 16, 32),
            channels=64,
            dw_kernel_size=None,
            layer_scale=None,
            offset_scale=1.0,
            mlp_ratio=4.0,
            post_norm=False,
            res_post_norm=False,
            level2_post_norm=False,
            level2_post_norm_block_ids=None,
            center_feature_scale=False,
            use_clip_projector=False,
            remove_center=False,
            num_classes=1000,
            drop_rate=0.0,
            drop_path_rate=0.0,
            drop_path_type='linear',
            act_layer='GELU',
            norm_layer='LN',
            cls_scale=1.5,
            with_cp=False,
            **kwargs,
    ):
        super().__init__(**kwargs)

        # Model configuration parameters
        self.core_op = core_op
        self.depths = depths
        self.groups = groups
        self.channels = channels
        self.dw_kernel_size = dw_kernel_size
        self.layer_scale = layer_scale
        self.offset_scale = offset_scale
        self.mlp_ratio = mlp_ratio
        self.post_norm = post_norm
        self.res_post_norm = res_post_norm
        self.level2_post_norm = level2_post_norm
        self.level2_post_norm_block_ids = level2_post_norm_block_ids
        self.center_feature_scale = center_feature_scale
        self.use_clip_projector = use_clip_projector
        self.remove_center = remove_center
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.drop_path_type = drop_path_type
        self.act_layer = act_layer
        self.norm_layer = norm_layer
        self.cls_scale = cls_scale
        self.with_cp = with_cp
