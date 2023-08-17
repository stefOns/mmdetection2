# _base_ = [
#    '../_base_/datasets/coco_instance.py',
#    ' ../_base_/datasets/coco_instance_semantic.py',
#    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
# ]
# model settings
model = dict(
    type='SOLOv2',
    backbone=dict(
        type='ResNet',
        depth=101,
        init_cfg=dict(checkpoint='torchvision://resnet101'),
        num_stages=4,
        out_indices=(0, 1, 2, 3),  # C2, C3, C4, C5
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        num_outs=5),
       
    mask_head=dict(
        type='MultiSOLOV2Head',
        #type='SOLOV2Head',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=512,  # 256,
        strides=[8, 8, 16, 32, 32],
        scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
        pos_scale=0.2,        
        num_grids=[40, 36, 24, 16, 12],
        #out_channels=256,
        cls_down_index=0,
        mask_feature_head=dict(
            # in_channels=256,
            out_channels=256,
            feat_channels=128,
            start_level=0,
            end_level=3,
            #num_classes=256,
            mask_stride=4,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)
        ),
        loss_mask=dict(type='DiceLoss', use_sigmoid=True, loss_weight=3.0),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        #fp16_enabled=True),
        headlist=[80,14,13]#),
    ),
    # training and testing settings
    # note: should be inside model, is deprecated outside...
    train_cfg = dict(),
    test_cfg = dict(
        nms_pre=500,
        score_thr=0.1,
        mask_thr=0.5,
        filter_thr=0.05,
        kernel='gaussian',  # gaussian/linear
        sigma=2.0,
        max_per_img=100)
)
	
# dataset settings
dataset_type = 'MultiDataset'
#classes = ('chair', 'couch', 'bed', 'dining table')
#data_root = 'data/coco/'
data_root = 'data/scannet/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=[(1333, 800), (1333, 768), (1333, 736), (1333, 704),
                   (1333, 672), (1333, 640)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800), #(2048,1024), #(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=0,#4,
    train=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/instances_train2017_furniture.json',
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        #classes=classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/instances_val2017_furniture.json',
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        #classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/instances_val2017_furniture.json',
        #ann_file=data_root + 'annotations/instances_val2017.json',
        ann_file=data_root + 'annotations/annotations_additional14classes/test-tol5-scannet-v2.json',
        img_prefix=data_root + 'test/',
        #img_prefix=data_root + 'val2017/',
        #classes=classes,
        pipeline=test_pipeline))
        
        
        
        
        
        

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[27, 33])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 36  # 36
device_ids = range(8)  # 8)
dist_params = dict(backend='nccl')
log_level = 'DEBUG'
work_dir = './work_dirs/solov2_r101_3x_multi'
load_from = None
resume_from = './work_dirs/solov2_r101_3x_multi/latest.pth'
#resume_from = './work_dirs/solov2_mmdet30_scannet_14additionalCocoBackbone_v2_0/latest.pth' #./work_dirs/solov2_r101_3x_scannet_20classes_visibilityConstraint_Refined/latest.pth'
workflow = [('train', 1)]
runner = dict(type='EpochBasedRunner', max_epochs=36)
evaluation = dict(interval=1, metric=['segm'], classwise=True)

