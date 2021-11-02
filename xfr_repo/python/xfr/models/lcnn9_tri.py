#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time    : 31/03/2020
# @Author  : Fangliang Bai
# @File    : lcnn9_tri.py
# @Software: PyCharm
# @Description:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from xfr.models.lightcnn import LightCNN_9Layers
# from xfr_repo.utils.utils import *

class LCNN9EmbeddingNet(nn.Module):
    def __init__(self):
        super(LCNN9EmbeddingNet, self).__init__()
        # self.name = os.path.splitext(os.path.split(__file__)[1])[0]
        self.reset()
        self.pool5_7x7_s1 = nn.AvgPool2d(kernel_size=[8, 8], stride=[1, 1], padding=0)
        # self.feat_extract = nn.Conv2d(128, 256, kernel_size=[1, 1], stride=(1, 1))

    def forward(self, x):
        self.features_part1.eval()
        # with torch.no_grad():
        x = self.features_part1(x)

        self.features_part2.eval()
        # with torch.no_grad():
        x = self.features_part2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        out = self.fc2(x)
        return out, x

    def reset(self):
        basenet = LightCNN_9Layers()

        self.features_part1 = nn.Sequential(
                *basenet.features[0:5]
        )
        self.features_part2 = nn.Sequential(
                *basenet.features[5:]
        )
        self.fc = basenet.fc1
        self.fc2 = basenet.fc2
        # fix_parameters(self.features_part1)
        # fix_parameters(self.features_part2)
