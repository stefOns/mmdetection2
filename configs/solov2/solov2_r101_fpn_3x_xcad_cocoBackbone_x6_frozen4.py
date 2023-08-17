_base_ = 'solov2_r50_fpn_3x_xcad_cocoBackbone.py'

# model settings
model = dict(
    backbone=dict(
        depth=101,
        frozen_stages=4,
        init_cfg=dict(checkpoint='torchvision://resnet101')
    ))
            
data_root = 'data/xCAD/'    
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        ann_file=data_root + 'coco_format/train/kfold_trainfiles_v3/xCAD_x6_train.json'),
    val=dict(
        ann_file=data_root + 'coco_format/val/v3/xCAD_x6_val.json'),
    test=dict(        
        ann_file=data_root + 'coco_format/val/v3/xCAD_x6_val.json'))
        
#resume_from = './work_dirs/models/xcad_v2_kfold/x6_frozen4/latest.pth'#'None
resume_from = None#'./work_dirs/models/xcad_v2_kfold/x6_frozen4/latest.pth'#'None
work_dir = './work_dirs/models/xcad_v2_kfold/x6_frozen4/'
#load_from = None
load_from = './checkpoints/SOLOv2_R101_3x_upgradedMMDET_init33classes_v4.pth' #None

evaluation = dict(metric=['segm'], classwise=True)
#evaluation = dict(interval=1, metric=['bbox', 'segm'], classwise=True)
