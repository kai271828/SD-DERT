_base_ = [
    '../_base_/datasets/aitod_detection.py',
    '../_base_/default_runtime.py'
]

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa

model = dict(
    type='DeformableDETR',
    num_queries=1500, # increased for aitod
    num_feature_levels=4,
    with_box_refine=True, # extra
    as_two_stage=True, # extra
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    ################################################################################################################################
    backbone=dict(
        # _delete_=True,
        type='SwinTransformer',
        pretrain_img_size=224,
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    ################################################################################################################################
    neck=dict(
        type='ChannelMapper',
        in_channels=[96, 192, 384, 768], # +1 channel
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    encoder=dict(  # DeformableDetrTransformerEncoder
        num_layers=6,
        layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
            self_attn_cfg=dict(  # MultiScaleDeformableAttention
                embed_dims=256,
                batch_first=True),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.1))),
    decoder=dict(  # DeformableDetrTransformerDecoder
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(  # DeformableDetrTransformerDecoderLayer
            self_attn_cfg=dict(  # MultiheadAttention
                embed_dims=256,
                num_heads=8,
                dropout=0.1,
                batch_first=True),
            cross_attn_cfg=dict(  # MultiScaleDeformableAttention
                embed_dims=256,
                batch_first=True),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.1)),
        post_norm_cfg=None),
    positional_encoding=dict(num_feats=128, normalize=True, offset=-0.5),
    bbox_head=dict(
        type='DeformableDETRHead',
        num_classes=8, # 8 classes in aitod
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='SmoothL1Loss', loss_weight=5.0), # change from L1Loss to SmoothL1loss
        loss_iou=dict(type='WassersteinLoss', loss_weight=2.0)), # change from GIoULoss to WassersteinLoss
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=1500)) # increased for aitod

# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
# train_pipeline = [
#     dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='RandomFlip', prob=0.5),
#     dict(
#         type='RandomChoice',
#         transforms=[
#             [
#                 dict(
#                     type='RandomChoiceResize',
#                     scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
#                             (608, 1333), (640, 1333), (672, 1333), (704, 1333),
#                             (736, 1333), (768, 1333), (800, 1333)],
#                     keep_ratio=True)
#             ],
#             [
#                 dict(
#                     type='RandomChoiceResize',
#                     # The radio of all image in train dataset < 7
#                     # follow the original implement
#                     scales=[(400, 4200), (500, 4200), (600, 4200)],
#                     keep_ratio=True),
#                 dict(
#                     type='RandomCrop',
#                     crop_type='absolute_range',
#                     crop_size=(384, 600),
#                     allow_negative_crop=True),
#                 dict(
#                     type='RandomChoiceResize',
#                     scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
#                             (608, 1333), (640, 1333), (672, 1333), (704, 1333),
#                             (736, 1333), (768, 1333), (800, 1333)],
#                     keep_ratio=True)
#             ]
#         ]),
#     dict(type='PackDetInputs')
# ]
# train_dataloader = dict(
#     dataset=dict(
#         filter_cfg=dict(filter_empty_gt=False), pipeline=train_pipeline))

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.0001),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))

# learning policy
max_epochs = 60
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[40],
        gamma=0.1)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (16 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=32)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=5),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))
