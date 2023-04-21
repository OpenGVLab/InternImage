custom_imports = dict(imports=['projects.openlanev2.baseline'])

method_para = dict(n_control=5) # #point for each curve

_dim_ = 128

model = dict(
    type='Baseline',
    img_backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[_dim_*2, _dim_*4],
        out_channels=_dim_,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='CustomIPMViewTransformer',
        num_cam=7,        
        xbound=[-50.0, 50.0, 1.0],
        ybound=[-25.0, 25.0, 1.0],
        zbound=[-3.0, 2.0, 0.5],
        out_channels=_dim_),
    lc_head=dict(
        type='CustomDETRHead',
        num_classes=1, 
        in_channels=_dim_,
        num_query=50,
        object_type='lane',
        num_layers=1,
        num_reg_dim=method_para['n_control']*3,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=2.5),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0), # dummy
        train_cfg=dict(
            assigner=dict(
                type='LaneHungarianAssigner',
                cls_cost=dict(type='FocalLossCost', weight=1.0),
                reg_cost=dict(type='LaneL1Cost', weight=2.5),
                iou_cost=dict(type='IoUCost', weight=0.0))), # dummy
        bev_range=[-50.0, -25.0, -3.0, 50.0, 25.0, 2.0]),
    te_head=dict(
        type='CustomDETRHead',
        num_classes=13, 
        in_channels=_dim_,
        num_query=30,
        object_type='bbox',
        num_layers=1,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=2.5),
        loss_iou=dict(type='GIoULoss', loss_weight=1.0),
        train_cfg=dict(
            assigner=dict(
                type='HungarianAssigner',
                cls_cost=dict(type='FocalLossCost', weight=1.0),
                reg_cost=dict(type='BBoxL1Cost', weight=2.5, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=1.0)))),
    lclc_head=dict(
        type='TopologyHead',
        in_channels=128,
        hidden_channels=_dim_,
        out_channels=1,
        num_layers=3,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0)),
    lcte_head=dict(
        type='TopologyHead',
        in_channels=128,
        hidden_channels=_dim_,
        out_channels=1,
        num_layers=3,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0)))

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

train_pipeline = [
    dict(type='CustomLoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='ResizeFrontView'),
    dict(type='CustomPadMultiViewImage', size_divisor=32),
    dict(type='CustomParameterizeLane', method='bezier_Endpointfixed', method_para=method_para),
    dict(type='CustomDefaultFormatBundle'),
    dict(
        type='Collect',
        keys=[
            'img',
            'gt_lc', 'gt_lc_labels',
            'gt_te', 'gt_te_labels',
            'gt_topology_lclc', 'gt_topology_lcte',
        ],
        meta_keys=[
            'scene_token', 'sample_idx', 'img_paths', 
            'img_shape', 'scale_factor', 'pad_shape',
            'lidar2img', 'can_bus',
        ],
    )
]
test_pipeline = [
    dict(type='CustomLoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='ResizeFrontView'),
    dict(type='CustomPadMultiViewImage', size_divisor=32),
    dict(type='CustomDefaultFormatBundle'),
    dict(
        type='Collect',
        keys=[
            'img',
        ],
        meta_keys=[
            'scene_token', 'sample_idx', 'img_paths', 
            'img_shape', 'scale_factor', 'pad_shape',
            'lidar2img', 'can_bus',
        ],
    )
]

dataset_type = 'OpenLaneV2SubsetADataset'
data_root = 'OpenLane-V2/data/OpenLane-V2'
meta_root = 'OpenLane-V2/data/OpenLane-V2'

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        meta_root=meta_root,
        collection='data_dict_subset_A_train',
        pipeline=train_pipeline,
        test_mode=False),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        meta_root=meta_root,
        collection='data_dict_subset_A_val',
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        meta_root=meta_root,
        collection='data_dict_subset_A_val',
        pipeline=test_pipeline,
        test_mode=True),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))

optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=1e-4)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

runner = dict(type='EpochBasedRunner', max_epochs=20)
evaluation = dict(interval=1, pipeline=test_pipeline)

checkpoint_config = dict(interval=1, max_keep_ckpts=1)

# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
