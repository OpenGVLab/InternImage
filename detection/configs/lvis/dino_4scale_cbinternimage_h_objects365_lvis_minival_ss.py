_base_ = [
    '../_base_/datasets/lvis_v1_instance_minival.py',
    '../_base_/default_runtime.py'
]
load_from = 'https://huggingface.co/OpenGVLab/InternImage/resolve/main/dino_4scale_cbinternimage_h_objects365_80classes.pth'
model = dict(
    type='CBDINO',
    backbone=dict(
        type='CBInternImage',
        core_op='DCNv3',
        channels=320,
        depths=[6, 6, 32, 6],
        groups=[10, 20, 40, 80],
        mlp_ratio=4.,
        drop_path_rate=0.5,
        norm_layer='LN',
        layer_scale=None,
        offset_scale=1.0,
        post_norm=False,
        dw_kernel_size=5,  # for InternImage-H/G
        res_post_norm=True,  # for InternImage-H/G
        level2_post_norm=True,  # for InternImage-H/G
        level2_post_norm_block_ids=[5, 11, 17, 23, 29],  # for InternImage-H/G
        center_feature_scale=True,  # for InternImage-H/G
        with_cp=True,
        out_indices=[(0, 1, 2, 3), (1, 2, 3)],
        init_cfg=None,
    ),
    neck=[dict(
        type='CBChannelMapper',
        in_channels=[640, 1280, 2560],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4)],
    bbox_head=dict(
        type='CBDINOHead',
        num_query=900,
        num_classes=1203,
        in_channels=2048,  # TODO
        sync_cls_avg_factor=True,
        as_two_stage=True,
        with_box_refine=True,
        dn_cfg=dict(
            type='CdnQueryGenerator',
            noise_scale=dict(label=0.5, box=1.0),  # 0.5, 0.4 for DN-DETR
            group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=1000)),
        transformer=dict(
            type='DinoTransformer',
            two_stage_num_proposals=900,
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        dropout=0.0),  # 0.1 for DeformDETR
                    feedforward_channels=2048,  # 1024 for DeformDETR
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=2048,
                        num_fcs=2,
                        ffn_drop=0.,
                        use_checkpoint=True,
                        act_cfg=dict(type='ReLU', inplace=True),),
                    ffn_dropout=0.0,  # 0.1 for DeformDETR
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DinoTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.0),  # 0.1 for DeformDETR
                        dict(
                            type='MultiScaleDeformableAttention',
                            num_levels=4,
                            embed_dims=256,
                            dropout=0.0),  # 0.1 for DeformDETR
                    ],
                    feedforward_channels=2048,  # 1024 for DeformDETR
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=2048,
                        num_fcs=2,
                        ffn_drop=0.,
                        use_checkpoint=True,
                        act_cfg=dict(type='ReLU', inplace=True),),
                    ffn_dropout=0.0,  # 0.1 for DeformDETR
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            temperature=20,
            normalize=True),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),  # 2.0 in DeformDETR
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0)),
        snip_cfg=dict(
            type='v3',
            weight=0.1),
        fed_loss_cfg=dict(
            use_fed_loss=True,
            fed_loss_num_classes=50,
            dataset_names='lvis_v1_train',
            freq_weight_power=0.5,
        )
    ),
    test_cfg=dict(max_per_img=300))  # TODO: Originally 100
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize',
         img_scale=[(2000, 600), (2000, 1200)],
         multiscale_mode='range',
         keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2000, 1000),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(dataset=dict(pipeline=train_pipeline)),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='AdamW', lr=0.0001, weight_decay=0.0001,
    constructor='CustomLayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=50, layer_decay_rate=0.94,
                       depths=[6, 6, 32, 6], offset_lr_scale=1e-3))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[])
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(interval=500, max_keep_ckpts=10)
evaluation = dict(interval=500)
# resume_from = None
# custom_hooks = [
#     dict(
#         type='ExpMomentumEMAHook',
#         resume_from=resume_from,
#         momentum=0.0003,
#         priority=49),
#     dict(
#         type='ZeroHook',
#         interval=500,
#         priority=49),
# ]
