# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
dataset_type = 'AITODDataset'
data_root = '/home/u2339555/AITOD/aitod/'

# img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
    ############################################################################################################################
    dict(type='Resize', scale=(800, 800), keep_ratio=True),
    ############################################################################################################################
    # dict(type='RandomFlip', flip_ratio=0.5),
    ############################################################################################################################
    dict(type='RandomFlip', prob=0.5),
    ############################################################################################################################
    # dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size_divisor=32),
    # dict(type='DefaultFormatBundle'),
    # dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
    ############################################################################################################################
    dict(type='PackDetInputs')
    ############################################################################################################################
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(800, 800),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
    ############################################################################################################################
    dict(type='Resize', scale=(800, 800), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape','scale_factor')
    )
    ############################################################################################################################
]

# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=2,
#     train=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/aitod_train.json',
#         img_prefix=data_root + 'train/',
#         pipeline=train_pipeline),
#     val=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/aitod_val.json',
#         img_prefix=data_root + 'val/',
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/aitod_test.json',
#         img_prefix=data_root + 'test/',
#         pipeline=test_pipeline))
################################################################################################################################
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/aitod_train.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/aitod_val.json',
        data_prefix=dict(img='val/'),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/aitod_test.json',
        data_prefix=dict(img='test/'),
        test_mode=True,
        pipeline=test_pipeline))
################################################################################################################################
# evaluation = dict(interval=12, metric='bbox')
################################################################################################################################
val_evaluator = dict(
    type='AITODMetric',
    ann_file=data_root + 'annotations/aitod_val.json',
    metric=['bbox'],
    format_only=False)
test_evaluator = dict(
    type='AITODMetric',
    ann_file=data_root + 'annotations/aitod_test.json',
    metric=['bbox'],
    format_only=False)
################################################################################################################################
