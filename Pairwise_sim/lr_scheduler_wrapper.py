# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 12:09:30 2020

@author: Sivapriyaa
"""
import torch.optim as optim
#from config import Configuration as C

C = None

def set_config(_C):
    global C
    C = _C
    
class LRSchedulerWrapper(object):
    
    def __init__(self, optimizer, lr_scheduler_type='StepLR', last_epoch=-1):
        self.lr_scheduler_type = lr_scheduler_type
        self._lr_scheduler = None
        self.optimizer = optimizer
        self.reset(last_epoch)
        
    def batch_step(self, epoch):
        if self.lr_scheduler_type in ['CosineAnnealingWarmRestarts']:
            self._lr_scheduler.step(epoch)
        
    def epoch_step(self, val_loss):
        if self.lr_scheduler_type == 'ReduceLROnPlateau':
            self._lr_scheduler.step(val_loss)
        elif self.lr_scheduler_type in ['StepLR', 'CosineAnnealingLR']:
            self._lr_scheduler.step()
            
    def reset(self, last_epoch=-1):
        lr_scheduler = \
            {'StepLR'                     : optim.lr_scheduler.StepLR(self.optimizer, step_size=C.lr_step_size, gamma=C.lr_decay_rate, last_epoch=last_epoch),
             'CosineAnnealingLR'          : optim.lr_scheduler.CosineAnnealingLR(self.optimizer, C.T_max, C.eta_min, last_epoch=last_epoch),
             'CosineAnnealingWarmRestarts': optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, C.T_0, C.T_mult, C.eta_min, last_epoch=last_epoch),
             'ReduceLROnPlateau'          : optim.lr_scheduler.ReduceLROnPlateau(self.optimizer),
            }
        if self.lr_scheduler_type in lr_scheduler:
            self._lr_scheduler = lr_scheduler[self.lr_scheduler_type]
        else:
            raise ValueError('lr scheduler type {} is not supported'.format(self.lr_scheduler_type))
        