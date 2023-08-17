import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, normal_init

from mmdet.models.builder import HEADS

from .mask_feat_head import MaskFeatHead

@HEADS.register_module()
class MultiMaskFeatHead(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 start_level,
                 end_level,
                 num_classes,
                 conv_cfg=None,
                 norm_cfg=None,
                 headlist=[]):
        #super(MultiMaskFeatHead, self).__init__(in_channels,
        #         out_channels,
        #         start_level,
        #         end_level,
        #         num_classes,
        #         conv_cfg,
        #         norm_cfg)

        
        super(MultiMaskFeatHead, self).__init__()

        self.headlist = headlist
                
        self.start_level = start_level
        self.end_level = end_level
                
        self.subheads = nn.ModuleList()

        for heads in self.headlist:
            maskfeathead = MaskFeatHead( in_channels,
                 out_channels,
                 start_level,
                 end_level,
                 num_classes,
                 conv_cfg,
                 norm_cfg)
            self.subheads.append(maskfeathead)

    def init_weights(self):
        for maskfeathead in self.subheads:
            maskfeathead.init_weights()   

    def forward(self, inputs):
        #print("MultiMaskFeatHead.forward")
        #print("MultiMaskFeatHead.forward - inputs: "+str(len(inputs)))

        #result = MaskFeatHead.forward(self, inputs)
        #print("MultiMaskFeatHead.forward - result: "+str(len(result)))
        
        #return result
        
        feature_list = []
        
        for i in range(len(self.subheads)):
        
            maskfeathead = self.subheads[i]
            
            feature_pred = maskfeathead.forward(inputs)
          
            
            feature_list.append(feature_pred)
        
        return (feature_list,)
