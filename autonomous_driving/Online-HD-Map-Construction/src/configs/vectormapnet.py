_base_ = [
    './_base_/default_runtime.py'
]

# meta info for submission file

meta = {
    'use_lidar': False,
    'use_camera': True,
    'use_external': False,
    'output_format': 'vector',

    # NOTE: please modify the information below
    'method': 'VectorMapNet', # name of your method
    'authors': ['Yicheng Liu', 'Tianyuan Yuan', 'Yue Wang',
        'Yilun Wang', 'Hang Zhao'], # author names
    'e-mail': 'yuantianyuan01@gmail.com', # your e-mail address
    'institution / company': 'MarsLab, Tsinghua University', # your organization
    'country / region': 'xxx', # (IMPORTANT) your country/region in iso3166 standard
}

# model type
type = 'Mapper'
plugin = True

# plugin code dir
plugin_dir = 'src/'

# img configs
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

img_size = (int(128*2), int((16/9*128)*2))

# category configs
cat2id = {
    'ped_crossing': 0,
    'divider': 1,
    'boundary': 2,
}
num_class = max(list(cat2id.values())) + 1

# bev configs
roi_size = (60, 30) # bev range, 60m in x-axis, 30m in y-axis
canvas_size = (200, 100) # bev feature size

# vectorize params
coords_dim = 2 # polylines coordinates dimension, 2 or 3
sample_dist = -1 # sampling params, vectormapnet uses simplify
sample_num = -1 # sampling params, vectormapnet uses simplify
simplify = True # sampling params, vectormapnet uses simplify

# model configs
head_dim = 256
norm_cfg = dict(type='BN2d')
num_points = 30

model = dict(
    type='VectorMapNet',
    backbone_cfg=dict(
        type='IPMEncoder',
        img_backbone=dict(
            type='ResNet',
            with_cp=False,
            pretrained='open-mmlab://detectron2/resnet50_caffe',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=-1,
            norm_cfg=norm_cfg,
            norm_eval=True,
            style='caffe',
            dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
            stage_with_dcn=(False, False, True, True)),
        img_neck=dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=128,
            start_level=0,
            add_extra_convs=True,
            # extra_convs_on_inputs=False,  # use P5
            num_outs=4,
            norm_cfg=norm_cfg,
            relu_before_extra_convs=True),
        upsample=dict(
            zoom_size=(1, 2, 4, 8),
            in_channels=128,
            out_channels=128,),
        xbound=[-roi_size[0]/2, roi_size[0]/2, roi_size[0]/canvas_size[0]],
        ybound=[-roi_size[1]/2, roi_size[1]/2, roi_size[1]/canvas_size[1]],
        heights=[-1.1, 0, 0.5, 1.1],
        out_channels=128,
        pretrained=None,
        num_cam=7,
        ),
    head_cfg=dict(
        type='DGHead',
        augmentation=True,
        augmentation_kwargs=dict(
            p=0.3,scale=0.01,
            bbox_type='xyxy',
            ),
        det_net_cfg=dict(
            type='MapElementDetector',
            num_query=120,
            max_lines=35,
            bbox_size=2,
            mode='xyxy',
            canvas_size=canvas_size,
            separate_detect=False,
            discrete_output=False,
            num_classes=num_class,
            in_channels=128,
            score_thre=0.1,
            num_reg_fcs=2,
            num_points=4,
            iterative=False,
            sync_cls_avg_factor=True,
            transformer=dict(
                type='DeformableDetrTransformer_',
                encoder=dict(
                    type='PlaceHolderEncoder',
                    embed_dims=head_dim,
                ),
                decoder=dict(
                    type='DeformableDetrTransformerDecoder_',
                    num_layers=6,
                    return_intermediate=True,
                    transformerlayers=dict(
                        type='DetrTransformerDecoderLayer',
                        attn_cfgs=[
                            dict(
                                type='MultiheadAttention',
                                embed_dims=head_dim,
                                num_heads=8,
                                attn_drop=0.1,
                                proj_drop=0.1,
                                dropout_layer=dict(type='Dropout', drop_prob=0.1),),
                            dict(
                                type='MultiScaleDeformableAttention',
                                embed_dims=head_dim,
                                num_heads=8,
                                num_levels=1,
                                ),
                        ],
                        ffn_cfgs=dict(
                            type='FFN',
                            embed_dims=head_dim,
                            feedforward_channels=head_dim*2,
                            num_fcs=2,
                            ffn_drop=0.1,
                            act_cfg=dict(type='ReLU', inplace=True),        
                        ),
                        feedforward_channels=head_dim*2,
                        ffn_dropout=0.1,
                        operation_order=('norm', 'self_attn', 'norm', 'cross_attn',
                                        'norm', 'ffn',)))
                ),
            positional_encoding=dict(
                type='SinePositionalEncoding',
                num_feats=head_dim//2,
                normalize=True,
                offset=-0.5),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0),
            loss_reg=dict(
                type='LinesLoss',
                loss_weight=0.1),
            train_cfg=dict(
                assigner=dict(
                    type='HungarianLinesAssigner',
                    cost=dict(
                        type='MapQueriesCost',
                        cls_cost=dict(type='FocalLossCost', weight=2.0),
                        reg_cost=dict(type='BBoxCostC', weight=0.1), # continues
                        iou_cost=dict(type='IoUCostC', weight=1,box_format='xyxy'), # continues
                        ),
                    ),
                ),
        ),
        gen_net_cfg=dict(
            type='PolylineGenerator',
            in_channels=128,
            encoder_config=None,
            decoder_config={
                    'layer_config': {
                        'd_model': 256,
                        'nhead': 8,
                        'dim_feedforward': 512,
                        'dropout': 0.2,
                        'norm_first': True,
                        're_zero': True,
                    },
                    'num_layers': 6,
                },
            class_conditional=True,
            num_classes=num_class,
            canvas_size=canvas_size, #xy
            max_seq_length=500,
            decoder_cross_attention=False,
            use_discrete_vertex_embeddings=True,
        ),
        max_num_vertices=80,
        top_p_gen_model=0.9,
        sync_cls_avg_factor=True,
        ),  
    with_auxiliary_head=False,
    model_name='VectorMapNet'
)

# data processing pipelines
train_pipeline = [
    dict(
        type='VectorizeMap',
        coords_dim=coords_dim,
        roi_size=roi_size,
        simplify=True,
        normalize=True,
    ),
    dict(
        type='PolygonizeLocalMapBbox',
        canvas_size=canvas_size,  # xy
        coord_dim=2,
        num_class=num_class,
        threshold=4/200,
    ),
    dict(type='LoadMultiViewImagesFromFiles'),
    dict(type='ResizeMultiViewImages',
         size = (int(128*2), int((16/9*128)*2)), # H, W
         change_intrinsics=True,
         ),
    dict(type='Normalize3D', **img_norm_cfg),
    dict(type='PadMultiViewImages', size_divisor=32, change_intrinsics=True),
    dict(type='FormatBundleMap'),
    dict(type='Collect3D', keys=['img', 'polys', 'vectors'], meta_keys=(
        'token', 'ego2img'))
]

test_pipeline = [
    dict(type='LoadMultiViewImagesFromFiles'),
    dict(type='ResizeMultiViewImages',
         size=img_size, # H, W
         change_intrinsics=True,
         ),
    dict(type='Normalize3D', **img_norm_cfg),
    dict(type='PadMultiViewImages', size_divisor=32, change_intrinsics=True),
    dict(type='FormatBundleMap'),
    dict(type='Collect3D', keys=['img'], meta_keys=(
        'token', 'ego2img'))
]

# dataset configs
data = dict(
    samples_per_gpu=5,
    workers_per_gpu=5,
    train=dict(
        type='AV2Dataset',
        ann_file='./data/train_annotations.json',
        root_path='./data/argoverse2/',
        meta=meta,
        roi_size=roi_size,
        cat2id=cat2id,
        pipeline=train_pipeline,
        interval=1,
    ),
    val=dict(
        type='AV2Dataset',
        ann_file='./data/val_annotations.json',
        root_path='./data/argoverse2/',
        meta=meta,
        roi_size=roi_size,
        cat2id=cat2id,
        pipeline=test_pipeline,
        test_mode=True,
        interval=1,
    ),
    test=dict(
        type='AV2Dataset',
        ann_file='./data/test_annotations.json',
        root_path='./data/argoverse2/',
        meta=meta,
        roi_size=roi_size,
        cat2id=cat2id,
        pipeline=test_pipeline,
        test_mode=True,
        interval=1,
    ),
)

# optimizer
optimizer = dict(
    type='AdamW',
    lr=1e-3,
    paramwise_cfg=dict(
    custom_keys={
        'backbone': dict(lr_mult=0.1),
    }),
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=3.5, norm_type=2))

# learning policy & schedule
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=400,
    warmup_ratio=0.1,
    step=[100, 120])
checkpoint_config = dict(interval=5)
total_epochs = 130

# kwargs for dataset evaluation
eval_kwargs = dict()
evaluation = dict(
    interval=5, 
    **eval_kwargs)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

find_unused_parameters = True
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
