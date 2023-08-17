#python modify_chkpt.py --src1 ../checkpoints/SOLOv2_R101_3x_upgraded.pth --target ../checkpoints/SOLOv2_R101_3x_upgraded_init18classes.pth --config ../configs/solov2/solov2_r101_3x_scannet_18classes.py --operation newhead

#python modify_chkpt.py --src1 ../checkpoints/epoch_28.pth --target ../checkpoints/solov2_coco_mmdetUpgraded.pth --config ../configs/solov2/solov2_r101_3x_scannet_20classes_cocoBackbone.py --operation upgrade_mmdet

#python modify_chkpt.py --src1 ../checkpoints/SOLOv2_R101_3x_upgraded.pth --target ../checkpoints/SOLOv2_R101_3x_upgraded_init20classesVisConstraint.pth --config ../configs/solov2/solov2_r101_3x_scannet_20classes_visibilityConstraint_Refined_10.py --operation newhead

#python ./tools/modify_chkpt.py --src1 checkpoints/20sclasses_scannetCocoLayer/epoch_28.pth  --target checkpoints/20classes_scannetCocoLayer/epoch_28_filteredto14.pth --remove_cls 3,4,6,14,16,17 --config configs/solov2/solov2_r101_3x_scannet_14classes_cocoBackbone.py

# filter classes from model - filtering overlap with COCO from Scannet 20
python ./tools/modify_chkpt.py --operation filter --src1 checkpoints/multi_pths/20classes_scannetCocoLayer/epoch_28.pth  --target checkpoints/multi_pths/20classes_scannetCocoLayer/epoch_28_filteredto14.pth --remove_cls 3,4,6,14,16,17 --config configs/solov2/solov2_r101_3x_scannet_20classes_cocoBackbone_v2_0.py


# merge multiple models to multi-head model
#python ./tools/modify_chkpt.py --operation multihead --srclist checkpoints/SOLOv2_R101_3x_upgraded.pth,checkpoints/20classes_scannetCocoLayer/epoch_28_filteredto14.pth --target checkpoints/SOLOv2_R101_3x_multi.pth --config configs/solov2/solov2_r101_3x_multi.py

## 12.07.2023 for multi for xcad
python ./tools/modify_chkpt.py --operation filter --src1 checkpoints/multi_pths/20classes_scannetCocoLayer/epoch_30.pth  --target checkpoints/multi_pths/20classes_scannetCocoLayer/epoch_30_filteredto14.pth --remove_cls 3,4,6,14,16,17 --config configs/solov2/solov2_r101_3x_scannet_20classes_cocoBackbone_v2_0.py
python ./tools/modify_chkpt.py --operation filter --src1 checkpoints/multi_pths/14AdditionalClasses_scannetCocoLayer/epoch_28.pth  --target checkpoints/multi_pths/14AdditionalClasses_scannetCocoLayer/epoch_28_filteredto13.pth --remove_cls 11 --config configs/solov2/solov2_r101_3x_scannet_14additionalclasses_cocoBackbone_v2_0.py

python ./tools/modify_chkpt.py --operation multihead --srclist checkpoints/multi_pths/SOLOv2_R101_3x_upgraded.pth,checkpoints/multi_pths/20classes_scannetCocoLayer/epoch_30_filteredto14.pth,checkpoints/multi_pths/14AdditionalClasses_scannetCocoLayer/epoch_28_filteredto13.pth --target checkpoints/multi_pths/v2_0/SOLOv2_R101_3x_multi_v2_0.pth --config configs/solov2/solov2_r101_3x_multi.py
