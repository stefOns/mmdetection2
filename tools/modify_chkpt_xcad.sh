#python modify_chkpt.py --src1 ../checkpoints/SOLOv2_R101_3x_upgraded.pth --target ../checkpoints/SOLOv2_R101_3x_upgraded_init18classes.pth --config ../configs/solov2/solov2_r101_3x_scannet_18classes.py --operation newhead
#python modify_chkpt.py --src1 ../checkpoints/twins_svt-l_uperhead_8x2_512x512_160k_ade20k_20211130_141005-3e2cae61.pth --target ../checkpoints/twins_init2classes.pth --config ../configs/twins/twins_svt-l_uperhead_8x2_512x512_160k_ade20k_floramon_veg.py --operation newhead
python modify_chkpt.py --src1 ../checkpoints/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K_20210531_125459-429057bf.pth --target ../checkpoints/swin_init33classes.pth --config ../configs/swin/upernet_swin_xcad_33classes.py --operation newhead


# filter classes from model - filtering overlap with COCO from Scannet 20
#python ./tools/modify_chkpt.py --operation filter --src1 checkpoints/20classes_scannetCocoLayer/epoch_28.pth  --target checkpoints/20classes_scannetCocoLayer/epoch_28_filteredto14.pth --remove_cls 3,4,6,14,16,17 --config configs/solov2/solov2_r101_3x_scannet_14classes_cocoBackbone.py
python ./tools/modify_chkpt.py --operation filter --src1 checkpoints/20classes_scannetCocoLayer/epoch_28.pth  --target checkpoints/20classes_scannetCocoLayer/epoch_28_filteredto14.pth --remove_cls 3,4,6,14,16,17 --config configs/solov2/solov2_r101_3x_scannet_14classes_cocoBackbone.py


# merge multiple models to multi-head model
python ./tools/modify_chkpt.py --operation multihead --srclist checkpoints/SOLOv2_R101_3x_upgraded.pth,checkpoints/20classes_scannetCocoLayer/epoch_28_filteredto14.pth --target checkpoints/SOLOv2_R101_3x_multi.pth --config configs/solov2/solov2_r101_3x_multi.py
