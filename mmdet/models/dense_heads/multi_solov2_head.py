import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init
from scipy import ndimage

#from mmdet.core import matrix_nms, multi_apply, mask_matrix_nms
from mmdet.core import InstanceData, mask_matrix_nms, multi_apply
from ..builder import HEADS, build_loss
from .solov2_head import SOLOV2Head

from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16, force_fp32

from mmdet.core.utils import center_of_mass, generate_coordinate
from mmdet.models.builder import HEADS
from mmdet.utils.misc import floordiv
from .solo_head import SOLOHead

@HEADS.register_module()
class MultiSOLOV2Head(nn.Module):

    """SOLOv2 mask head used in `SOLOv2: Dynamic and Fast Instance
    Segmentation. <https://arxiv.org/pdf/2003.10152>`_

    Args:
        mask_feature_head (dict): Config of SOLOv2MaskFeatHead.
        dynamic_conv_size (int): Dynamic Conv kernel size. Default: 1.
        dcn_cfg (dict): Dcn conv configurations in kernel_convs and cls_conv.
            default: None.
        dcn_apply_to_all_conv (bool): Whether to use dcn in every layer of
            kernel_convs and cls_convs, or only the last layer. It shall be set
            `True` for the normal version of SOLOv2 and `False` for the
            light-weight version. default: True.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 *args,
                 mask_feature_head,
                 dynamic_conv_size=1,
                 dcn_cfg=None,
                 dcn_apply_to_all_conv=True,
                 init_cfg=[
                     dict(type='Normal', layer='Conv2d', std=0.01),
                     dict(
                         type='Normal',
                         std=0.01,
                         bias_prob=0.01,
                         override=dict(name='conv_cls'))
                 ],
                 headlist = [],
                 **kwargs):
        assert dcn_cfg is None or isinstance(dcn_cfg, dict)
        self.dcn_cfg = dcn_cfg
        self.with_dcn = dcn_cfg is not None
        self.dcn_apply_to_all_conv = dcn_apply_to_all_conv
        self.dynamic_conv_size = dynamic_conv_size
        mask_out_channels = mask_feature_head.get('out_channels')
        self.kernel_out_channels = \
            mask_out_channels * self.dynamic_conv_size * self.dynamic_conv_size

        super(MultiSOLOV2Head,self).__init__()
        self.headlist = headlist
        self.subheads = nn.ModuleList() # headlist # TODO or inherit from BaseMaskHead???
        self.num_classes = 0
        for clsnum in self.headlist:
            print(clsnum)
            kwargs["num_classes"] = clsnum
            solov2head = SOLOV2Head(mask_feature_head=mask_feature_head, init_cfg=init_cfg, **kwargs)
            self.subheads.append(solov2head)
            self.num_classes = self.num_classes + clsnum

    def init_weights(self):
        for solov2head in self.subheads:
            solov2head.init_weights()        
                 
    def forward(self, feats, eval=False):
               
        print("MultiSOLOv2Head.forward - feats: "+str(len(feats)))
        #result = solov2head.forward(self, feats, eval)


        #return result
        
        mlvl_cls_preds_list = []
        mlvl_kernel_preds_list = []
        feature_pred_list = []
        mask_feats_list = []

        for solov2head in self.subheads:
            #cate_pred, kernel_pred = solov2head.forward(feats, eval)
            #feature_pred = solov2head.forward(feats)
            mlvl_kernel_preds, mlvl_cls_preds, mask_feats = solov2head.forward(feats)
            # result = feature_pred
            # print("MultiSOLOv2Head.forward - result: " + str(len(result)))
            # print(len(result[0]))
            # for i in range(len(result[0])):
            #     print(result[0][i].shape)
            # print(len(result[1]))
            # for i in range(len(result[1])):
            #     print(result[1][i].shape)

            #cate_list.append(cate_pred)
            mlvl_cls_preds_list.append(mlvl_cls_preds)
            #kernel_list.append(kernel_pred)
            mlvl_kernel_preds_list.append(mlvl_kernel_preds)
            mask_feats_list.append(mask_feats)

        return mlvl_kernel_preds_list, mlvl_cls_preds_list, mask_feats_list #feature_pred_list #cate_list, kernel_list


    #def forward_single(self, x, idx, eval=False, upsampled_size=None):
    #    print("MultiSOLOv2Head.forward_single - x: "+str(x.shape)+" idx: "+str(idx))
    #    result = SOLOv2Head.forward_single(self, x, idx, eval, upsampled_size)
    #    print("MultiSOLOv2Head.forward_single - result: "+str(len(result)))
    #    print(result[0].shape)
    #    print(result[1].shape)
        
    #    return result        

    def get_seg(self,
                cate_preds,
                kernel_preds,
                seg_pred,
                img_metas,
                cfg,
                rescale=None):
        
        
        #print("MultiSOLOv2Head.get_seg - cate_preds: "+str(len(cate_preds))+" kernel_preds: "+str(len(kernel_preds))+" seg_pred: "+str(seg_pred.shape)+" img_metas: "+str(len(img_metas)))
        #result = SOLOv2Head.get_seg(self, cate_preds, kernel_preds, seg_pred, img_metas, cfg, rescale)
        #print("MultiSOLOv2Head.get_seg - result: "+str(len(result)))
 
        all_bbox_result_list = []
        all_segm_result_list = []
        all_result_list = []
        
        
        offset = 0
        tmp = len(self.subheads)
        for i in range(len(self.subheads)):   #range(0,1):
        
            cate_predsI = cate_preds[i]
            kernel_predsI = kernel_preds[i]
            seg_predI = seg_pred[0][i]
          
            
            
            solov2head = self.subheads[i]
            
            bbox_result_list, segm_result_list, result_list = solov2head.get_seg(cate_predsI, kernel_predsI, seg_predI, img_metas, cfg, rescale)

            #print("head "+str(i))
            #print(len(result_list))
            #if result_list[0] !=None:
            #    print(len(result_list[0][0]))
            #    print(result_list[0])
            #    print(len(bbox_result_list[0][0]))
            #    print(bbox_result_list)


            for j in range(len(result_list)):
                # get the device we are on
                mydev = cate_predsI[0].device
            
                if result_list[j] == None:
                    result_list[j] = [ torch.from_numpy(np.array([],dtype=bool)).to(mydev), 
                                       torch.from_numpy(np.array([],dtype=int)).to(mydev), 
                                       torch.from_numpy(np.array([],dtype=np.float32)).to(mydev) ]
              
            # adjust class labels
            for r in result_list:
                if (len(r[1])>0):
                    r[1][:] = r[1][:] + offset

            if len(all_result_list)==0:
                all_bbox_result_list = bbox_result_list
                all_segm_result_list = segm_result_list
                all_result_list = result_list
            else:
                for j in range(len(all_result_list)):

                    all_bbox_result_list[j].extend(bbox_result_list[j])
                    all_segm_result_list[j].extend(segm_result_list[j])

                    all_result_list[j] = ( torch.cat((all_result_list[j][0],result_list[j][0]),0),
                                           torch.cat((all_result_list[j][1],result_list[j][1]),0),
                                           torch.cat((all_result_list[j][2],result_list[j][2]),0) )
                                
            
            offset = offset + self.headlist[i]
            
            
        #print(len(all_result_list[0]))
        #print(len(all_bbox_result_list[0]))
        #print(len(all_segm_result_list[0]))
        
        return all_bbox_result_list, all_segm_result_list, all_result_list        


    def simple_test(self,
                    feats,
                    img_metas,
                    rescale=False,
                    instances_list=None,
                    **kwargs):
        """Test function without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.
            instances_list (list[obj:`InstanceData`], optional): Detection
                results of each image after the post process. Only exist
                if there is a `bbox_head`, like `YOLACT`, `CondInst`, etc.

        Returns:
            list[obj:`InstanceData`]: Instance segmentation \
                results of each image after the post process. \
                Each item usually contains following keys. \

                - scores (Tensor): Classification scores, has a shape
                  (num_instance,)
                - labels (Tensor): Has a shape (num_instances,).
                - masks (Tensor): Processed mask results, has a
                  shape (num_instances, h, w).
        """
        if instances_list is None:
            outs = self(feats)
        else:
            outs = self(feats, instances_list=instances_list)
        mask_inputs = outs + (img_metas, )
        results_list = self.get_results(
            *mask_inputs,
            rescale=rescale,
            instances_list=instances_list,
            **kwargs)
        return results_list


    def get_results(self, mlvl_kernel_preds, mlvl_cls_scores, mask_feats,
                    img_metas, **kwargs):
        """Get multi-image mask results.

        Args:
            mlvl_kernel_preds (list[Tensor]): Multi-level dynamic kernel
                prediction. The kernel is used to generate instance
                segmentation masks by dynamic convolution. Each element in the
                list has shape
                (batch_size, kernel_out_channels, num_grids, num_grids).
            mlvl_cls_scores (list[Tensor]): Multi-level scores. Each element
                in the list has shape
                (batch_size, num_classes, num_grids, num_grids).
          def   mask_feats (Tensor): Unified mask feature map used to generate
                instance segmentation masks by dynamic convolution. Has shape
                (batch_size, mask_out_channels, h, w).
            img_metas (list[dict]): Meta information of all images.

        Returns:
            list[:obj:`InstanceData`]: Processed results of multiple
            images.Each :obj:`InstanceData` usually contains
            following keys.

                - scores (Tensor): Classification scores, has shape
                  (num_instance,).
                - labels (Tensor): Has shape (num_instances,).
                - masks (Tensor): Processed mask results, has
                  shape (num_instances, h, w).
        """

        # todo: check code from multi solov2 head py: digs113: fct: get_seg()
        # combine now all results from the heads:
        # go thoroug all subheads: call solov2_head.getseg() for each haead ---


        offset = 0

        all_result_list = []
        for i in range(len(self.subheads)):
            result_list = []

            mlvl_kernel_predsI = mlvl_kernel_preds[i]
            mlvl_cls_scoresI = mlvl_cls_scores[i]
            mask_featsI = mask_feats[i]

            num_levels = len(mlvl_cls_scoresI)
            assert len(mlvl_kernel_predsI) == len(mlvl_cls_scoresI)

            for lvl in range(num_levels):
                cls_scores = mlvl_cls_scoresI[lvl]
                cls_scores = cls_scores.sigmoid()
                local_max = F.max_pool2d(cls_scores, 2, stride=1, padding=1)
                keep_mask = local_max[:, :, :-1, :-1] == cls_scores
                cls_scores = cls_scores * keep_mask
                mlvl_cls_scoresI[lvl] = cls_scores.permute(0, 2, 3, 1)



            assert len(img_metas) == 1

            solov2head = self.subheads[i]

            for img_id in range(len(img_metas)):
                img_cls_pred = [
                    mlvl_cls_scoresI[lvl][img_id].view(-1, solov2head.cls_out_channels)
                    for lvl in range(num_levels)
                ]
                img_mask_feats = mask_featsI[[img_id]]
                img_kernel_pred = [
                    mlvl_kernel_predsI[lvl][img_id].permute(1, 2, 0).view(
                        -1, self.kernel_out_channels) for lvl in range(num_levels)
                ]
                img_cls_pred = torch.cat(img_cls_pred, dim=0)
                img_kernel_pred = torch.cat(img_kernel_pred, dim=0)

                result = solov2head._get_results_single(
                    img_kernel_pred,
                    img_cls_pred,
                    img_mask_feats,
                    img_meta=img_metas[img_id])

                result_list.append(result)

            for j in range(len(result_list)):
                mydev = mlvl_kernel_predsI[0].device

                if result_list[j] == None:
                    result_list[j] = [torch.from_numpy(np.array([],dtype=bool)).to(mydev),
                                      torch.from_numpy(np.array([],dtype=int)).to(mydev),
                                      torch.from_numpy(np.array([],dtype=np.float32)).to(mydev)]


            #adjust class labels
            for r in result_list:
                if (len(r.labels) > 0):
                    r.labels[:] = r.labels[:] + offset

            if len(all_result_list) == 0:
                #all_bbox_result_list = bbox_result_list
                #all_segm_result_list = segm_result_list
                all_result_list = result_list
            else:
                for j in range(len(all_result_list)):

                    # #all_bbox_result_list[j].extend(bbox_result_list[j])
                    # all_result_list[j] = (torch.cat((all_result_list[j][0], result_list[j][0]), 0),
                    #                       torch.cat((all_result_list[j][1], result_list[j][1]), 0),
                    #                       torch.cat((all_result_list[j][2], result_list[j][2]), 0))

                    #all_bbox_result_list[j].extend(bbox_result_list[j])
                    # all_result_list[j].labels.extend(result_list[j].labels)
                    # all_result_list[j].masks.extend(result_list[j].masks)
                    # all_result_list[j].scores.extend(result_list[j].scores)
                    if len(result_list[j].labels) > 0:
                        all_result_list[j].labels = torch.cat((all_result_list[j].labels, result_list[j].labels), 0)
                        all_result_list[j].masks = torch.cat((all_result_list[j].masks, result_list[j].masks), 0)
                        all_result_list[j].scores = torch.cat((all_result_list[j].scores, result_list[j].scores), 0)

            offset = offset + self.headlist[i]



        return all_result_list