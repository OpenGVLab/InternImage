# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
_base_ = [
    '../_base_/models/upernet_r50.py', '../_base_/datasets/mapillary.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
pretrained = 'https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_xl_22k_192to384.pth'
model = dict(
    backbone=dict(
        _delete_=True,
        type='InternImage',
        core_op='DCNv3',
        channels=192,
        depths=[5, 5, 24, 5],
        groups=[12, 24, 48, 96],
        mlp_ratio=4.,
        drop_path_rate=0.4,
        norm_layer='LN',
        layer_scale=1.0,
        offset_scale=2.0,
        post_norm=True,
        with_cp=False,
        out_indices=(0, 1, 2, 3),
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    decode_head=dict(num_classes=150, in_channels=[192, 384, 768, 1536]),
    auxiliary_head=dict(num_classes=150, in_channels=768),
    test_cfg=dict(mode='whole'))
optimizer = dict(
    _delete_=True, type='AdamW', lr=0.00002, betas=(0.9, 0.999), weight_decay=0.05,
    constructor='CustomLayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=39, layer_decay_rate=0.94,
                       depths=[5, 5, 24, 5], offset_lr_scale=1.0))
lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)
runner = dict(type='IterBasedRunner')
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=0.1, norm_type=2))
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=1)
evaluation = dict(interval=8000, metric='mIoU', save_best='mIoU')
# fp16 = dict(loss_scale=dict(init_scale=512))
