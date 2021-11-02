# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 14:11:25 2020

@author: Sivapriyaa
"""

import torch
import torch.nn as nn
from pretrained_models.light_cnn import LightCNN_9Layers as get_lcnn9_model, set_config as set_config_lcnn
from utils import fix_parameters
import os

C = None

def set_config(_C):
    global C
    C = _C
    set_config_lcnn(C)
    
class LCNN9EmbeddingNet(nn.Module):
    def __init__(self):      
        super(LCNN9EmbeddingNet, self).__init__()    
        self.name = os.path.splitext(os.path.split(__file__)[1])[0]
        self.reset()        
                
    def forward(self, x):        
        if type(x) in [tuple, list]:
            x, labels = x
#        timg=x
        x=self.basenet(x)
        return x
        
    def get_embedding(self, x):
        if type(x) in (tuple, list): # bypass labels if they are in the input list
            x = x[0]
        if self.training:
            self.eval()
        with torch.no_grad():
             x = self.basenet(x)
        return x
          
    def reset(self):
        self.basenet = get_lcnn9_model()
    
    
def get_model():
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    model = LCNN9EmbeddingNet()
    if C.lcnn9_tri_pth:
        state_dict = torch.load(C.lcnn9_tri_pth)
        model.load_state_dict(state_dict)
    return model
