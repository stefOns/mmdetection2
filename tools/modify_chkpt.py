import torch

import argparse
import os
import sys

import mmcv
from mmcv import Config, DictAction

from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
import numpy as np
from mmdet.utils import (collect_env, get_device, get_root_logger,
                         replace_cfg_vals, rfnext_init_model,
                         setup_multi_processes, update_data_root)

def parse_args():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument('--src1', type=str, default='',
                        help='Path to the main checkpoint')

    parser.add_argument('--srclist', type=str, default='',
                        help='Comma separated list of checkpoints to merge')

    parser.add_argument('--target', type=str, default='',
                        help='Name of the new ckpt')
    # operation
    parser.add_argument('--operation', choices=['newhead','adaptLabelSize','multihead','filter','upgrade_mmdet'],
                        required=True,
                        help='Operation to perform. newhead = new SOLO head with specified number of classes, multihead = merge heads of multiple checkpoints, filter = filter classes from checkpoint')

    parser.add_argument('--remove_cls', type=str, default='',
                        help='For filter operation: Comma separated list of class indices to remove from model')


    # Config
    parser.add_argument('--config', type=str, default='',
                        help='Configuration file for the target model')
                        
    args = parser.parse_args()
    return args


def upgrade_mmdet(args):
    cfg = Config.fromfile(args.config)

    # build new model
    model = build_detector(
        cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))

    # Load checkpoint
    ckpt = torch.load(args.src1)

    print(ckpt.keys())

    if 'model_state_dict' in ckpt.keys():
        state_dict = ckpt['model_state_dict']
        modified_state_dict = ckpt['model_state_dict'].copy()
    else:
        state_dict = ckpt['state_dict'].copy()
        modified_state_dict = ckpt['state_dict'].copy()

    # remove head parameters from checkpoint
    for key, value in state_dict.items():
        print(key)
        if key.startswith('bbox_head.cate_convs'):
            modified_state_dict[key] = 'mask_head.cls_convs'
            print("mask_head.cls_convs: "+str(value))
        if key.startswith('bbox_head.kernel_convs'):
            modified_state_dict[key] = 'mask_head.kernel_convs'
            print("mask_head.kernel_convs: " + str(value))
        if key.startswith('bbox_head.solo_cate'):
            modified_state_dict[key] = 'mask_head.conv_cls'
            print("mask_head.conv_cls: " + str(value))
        if key.startswith('bbox_head.solo_kernel'):
            modified_state_dict[key] = 'mask_head.conv_kernel'
            print("mask_head.conv_kernel: " + str(value))
        if key.startswith('mask_feat_head.'):
            modified_state_dict[key] = 'mask_head.mask_feature_head.'
            print("mask_head.mask_feature_head: " + str(value))
    # assign state dict to model (ignore missing keys)
    model.load_state_dict(modified_state_dict, False)

    save_path = args.target

    for key, value in model.state_dict().items():
        print(key)
        print(value.shape)

    # Save to file
    modified_ckpt = {}
    modified_ckpt['state_dict'] = model.state_dict()
    # modified_ckpt['model_state_dict'] = model.state_dict()
    save_ckpt(modified_ckpt, save_path)


def newhead(args):
    cfg  = Config.fromfile(args.config)

    # build new model
    model = build_detector(
        cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))

    # Load checkpoint
    ckpt = torch.load(args.src1)
    reset_ckpt(ckpt)

    print(ckpt.keys())

    if 'model_state_dict' in ckpt.keys():
        state_dict = ckpt['model_state_dict']
        modified_state_dict = ckpt['model_state_dict'].copy()
    else:
        state_dict = ckpt['state_dict'].copy()
        modified_state_dict = ckpt['state_dict'].copy()
    

    # remove head parameters from checkpoint
    for key, value in state_dict.items():
        print(key)
        print(value.shape)
        if key.startswith('roi_head.bbox_head') or key.startswith('roi_head.mask_head') or key.startswith('mask_head'):
            del modified_state_dict[key]

    # assign state dict to model (ignore missing keys)
    model.load_state_dict(modified_state_dict,False)

    save_path = args.target

    for key, value in model.state_dict().items():
        print(key)
        print(value.shape)
        
    

    # Save to file
    modified_ckpt = {}
    modified_ckpt['state_dict'] = model.state_dict()
    #modified_ckpt['model_state_dict'] = model.state_dict()
    save_ckpt(modified_ckpt, save_path)

    test_ckpt = torch.load(save_path)
    finish = 1

def adaptLabelSize(args):
    cfg = Config.fromfile(args.config)

    # build new model
    model = build_detector(
        cfg.model, train_cfg=cfg.model.train_cfg, test_cfg=cfg.model.test_cfg)

    # Load checkpoint
    ckpt = torch.load(args.src1)
    reset_ckpt(ckpt)

    print(ckpt.keys())

    if 'model_state_dict' in ckpt.keys():
        state_dict = ckpt['model_state_dict']
        modified_state_dict = ckpt['model_state_dict'].copy()
    else:
        state_dict = ckpt['state_dict'].copy()
        modified_state_dict = ckpt['state_dict'].copy()

    # remove head parameters from checkpoint
    for key, value in state_dict.items():
        if key.startswith('bbox_head'):
            del modified_state_dict[key]

    # assign state dict to model (ignore missing keys)
    model.load_state_dict(modified_state_dict, False)

    save_path = args.target

    for key, value in model.state_dict().items():
        print(key)
        print(value.shape)

    # Save to file
    modified_ckpt = {}
    modified_ckpt['state_dict'] = model.state_dict()
    # modified_ckpt['model_state_dict'] = model.state_dict()
    save_ckpt(modified_ckpt, save_path)



def multihead(args):

    cfg = Config.fromfile(args.config)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    #cfg_new = ""
    # build new model
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))


    chkpt_names = args.srclist.split(',')
    chkpt_list = []

    ## Load checkpoint
    #ckpt = load_checkpoint(model, args.src1, map_location='cpu')
    #reset_ckpt(ckpt)



    
    
    for cn in chkpt_names:
        ckpt = torch.load(cn)
        chkpt_list.append(ckpt)

    if 'model_state_dict' in chkpt_list[0].keys():
        state_dict = chkpt_list[0]['model_state_dict']
        modified_state_dict = chkpt_list[0]['model_state_dict'].copy()
    else:
        state_dict = chkpt_list[0]['state_dict']
        modified_state_dict = chkpt_list[0]['state_dict'].copy()
    
    # modify keys in first dict
    for key, value in state_dict.items():
        #if key.startswith('bbox_head') or key.startswith('mask_feat_head'):
        if key.startswith('bbox_head') or key.startswith('mask_feat_head') or key.startswith('mask_head'):
            del modified_state_dict[key]
            keyparts = key.split('.')
            newkey = keyparts[0]+'.subheads.0.'+'.'.join(keyparts[1:])
            modified_state_dict[newkey] = value
    
    # add head keys from other dicts
    for i in range(1,len(chkpt_list)):
        if 'model_state_dict' in chkpt_list[i].keys():
            state_dict = chkpt_list[i]['model_state_dict']
        else:
            state_dict = chkpt_list[i]['state_dict']
    
        for key, value in state_dict.items():
            #if key.startswith('bbox_head') or key.startswith('mask_feat_head'):
            if key.startswith('mask_head'):
                keyparts = key.split('.')
                newkey = keyparts[0]+'.subheads.'+str(i)+'.'+'.'.join(keyparts[1:])
                modified_state_dict[newkey] = value       

    # assign state dict to model (ignore missing keys)
    #model.load_state_dict(modified_state_dict,False)

    save_path = args.target

    for key, value in model.state_dict().items():
        print(key)
        print(value.shape)
        
    

    # Save to file
    modified_ckpt = chkpt_list[0]
    modified_ckpt['state_dict'] = modified_state_dict#model.state_dict()
    #modified_ckpt['model_state_dict'] = model.state_dict()
    save_ckpt(modified_ckpt, save_path)



def filterclasses(args):
    cfg = Config.fromfile(args.config)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    #cfg_new = ""
    # build new model
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    # Load checkpoint
    ckpt = load_checkpoint(model, args.src1, map_location='cpu')
    reset_ckpt(ckpt)

    if 'model_state_dict' in ckpt.keys():
        state_dict = ckpt['model_state_dict']
        modified_state_dict = ckpt['model_state_dict'].copy()
    else:
        state_dict = ckpt['state_dict'].copy()
        modified_state_dict = ckpt['state_dict'].copy()  

    # prepare class filter
    removecls = args.remove_cls
    clsindex = removecls.split(',')
    clsindex = [int(i) for i in clsindex]

    # change class list
    for key, value in state_dict.items():
        print(key)
        if key=='mask_head.conv_cls.weight': #key=='bbox_head.solo_cate.weight':

            keep = np.ones((value.shape[0],),dtype=int)
            keep[clsindex] = 0
            keep = np.nonzero(keep)[0]
            value = value[keep,:,:,:]
        elif key=='mask_head.conv_cls.bias': #key=='bbox_head.solo_cate.bias':
            keep = np.ones((value.shape[0],),dtype=int)
            keep[clsindex] = 0          
            keep = np.nonzero(keep)[0]
            value = value[keep]
        modified_state_dict[key] = value


    # assign state dict to model (ignore missing keys)
    #model_new.load_state_dict(modified_state_dict,False)

    save_path = args.target

    for key, value in model.state_dict().items():
        print(key)
        print(value.shape)
        
    

    # Save to file
    modified_ckpt = ckpt
    modified_ckpt['state_dict'] = modified_state_dict #model.state_dict()
    #modified_ckpt['model_state_dict'] = model.state_dict()
    save_ckpt(modified_ckpt, save_path)


def save_ckpt(ckpt, save_name):
    torch.save(ckpt, save_name)
    print('save changed ckpt to {}'.format(save_name))


def reset_ckpt(ckpt):
    if 'scheduler' in ckpt:
        del ckpt['scheduler']
    if 'optimizer' in ckpt:
        del ckpt['optimizer']
    if 'iteration' in ckpt:
        ckpt['iteration'] = 0



def main(arglist):

    sys.argv = arglist
    sys.argc = len(arglist)

    args = parse_args()


    if args.operation == 'newhead':
        newhead(args)
    elif args.operation == 'adaptLabelSize':
        adaptLabelSize(args)
    elif args.operation == 'multihead':
        multihead(args)
    elif args.operation == 'filter':
        filterclasses(args)
    elif args.operation == 'upgrade_mmdet':
        upgrade_mmdet(args)
    else:
        print('Unknown operation: '+args.operation)
        exit()
        
        
if __name__ == '__main__':
    
    main(sys.argv)
    
