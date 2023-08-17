import itertools
import logging
import os.path as osp
import tempfile
import json
import mmcv
import numpy as np
from mmcv.utils import print_log
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# from boundary_iou.coco_instance_api.coco import COCO
# from boundary_iou.coco_instance_api.cocoeval import COCOeval

from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .builder import DATASETS
from .custom import CustomDataset
from .coco import CocoDataset

import pycocotools

@DATASETS.register_module()
class xCADDataset(CocoDataset):
    CLASSES = (
        'book',
        'cabinet',
        'carpet',
        'ceiling',
        'chair',
        'counter',
        'curtain',
        'door',
        'floor',
        'lamp',
        'light',
        'microwave',
        'mirror',
        'oven',
        'pillow',
        'plant',
        'radiator',
        'refrigerator',
        'seating',
        'shelf',
        'shower',
        'sink',
        'sofa',
        'stairs',
        'stool',
        'switch',
        'table',
        'television stand',
        'tiled stove',
        'tv',
        'wall',
        'wall mounted coat rack',
        'window'
        )
    PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
               (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
               (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
               (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
               (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
               (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
               (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
               (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
               (78, 180, 255)]
