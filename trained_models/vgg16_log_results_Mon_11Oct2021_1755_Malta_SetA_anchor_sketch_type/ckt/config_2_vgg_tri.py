# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 12:09:30 2020

@author: Sivapriyaa
"""

from pathlib import Path
import sys
import os
import torch
from datetime import datetime
""" Configuration
    Defines all settings that configure the model training and testing.
    This abstract class provide an template of default configurations.
"""


class Configuration(object):
    # <editor-fold desc="[+] GPU Device ...">
    device_id = '0'
    cuda = torch.cuda.is_available()
    if cuda:
        device = torch.device('cuda:' + device_id)
    else:
        device = torch.device('cpu')     
    # </editor-fold>
    
    # <editor-fold desc="[+] Project Path ...">
    project_dir = Path(__file__).resolve().parents[0]
    root_dir = Path(__file__).resolve().parents[1]
    #*******************************
    dataset_dir=Path(__file__).resolve().parents[5] / 'Composite2Photo'
    #*******************************
    # </editor-fold>  
    
    # <editor-fold desc="[+] Pre-trained Model">
    vgg_face_dag_pth = project_dir / 'pretrained_models' / 'vgg_face_dag.pth'  # weights of the pre-trained model
#    senet50_256_pth = project_dir / 'pre_trained_models' / 'senet50_256.pth'
#    lcnn9_pth =project_dir / 'pre_trained_models' /'LightCNN_9Layers_checkpoint.pth.tar' 
    # </editor-fold>

   # <editor-fold desc="[+] Summury Path">
    date = datetime.now()
    date_time=date.strftime("%a_%d%b%Y_%H%M")
     
    train_summary_dir = project_dir / os.path.join('log_results', 'log_results_'+date_time) /'train_val_test_summary'/ 'train_summary'
    valid_summary_dir = project_dir / os.path.join('log_results', 'log_results_'+date_time) /'train_val_test_summary'/ 'valid_summary'
    test_summary_dir = project_dir / os.path.join('log_results', 'log_results_'+date_time) /'train_val_test_summary'/ 'test_summary'
     
    vis_dir = project_dir / 'Visualization'/'Results' 

    # </editor-fold>

    # <editor-fold desc="[+] Trained Model Path">
    restore_checkpoint = False # for testing set to True # If set true, you need to set checkpoint_file below to a valid path
    visualize = False
    
    if restore_checkpoint == False:
            trained_model_dir = project_dir / os.path.join('log_results', 'log_results_'+ date_time) / 'ckt'
            valid_reset_label=True # To reset label for validation classes # Set to True While calculating validation accuracies
    else:
        trained_model_dir = project_dir / os.path.join('log_results', 'log_results_Mon_11Oct2021_1043_Malta_SetA_anchor_sketch_type') / 'ckt'
        # trained_model_dir = project_dir / os.path.join('log_results', 'log_results_Tue_21Sep2021_1806_Malta_SetA_anchor_sketch_type') / 'ckt'
        valid_reset_label=False # To reset label for validation classes # Set to True While calculating validation accuracies
        
    if restore_checkpoint == False:
        os.makedirs(train_summary_dir)
        os.makedirs(valid_summary_dir)
        os.makedirs(test_summary_dir)
        os.makedirs(trained_model_dir)
    
    log_fname=os.path.join('log_results_'+ date_time)
    config_fname=os.path.basename(__file__) 
    config_fname_=config_fname.replace('.py','')
    save_results_dir= project_dir / 'log_results'/os.path.join('log_results_'+date_time)
    load_setB = False
    reset_label = True # Applicable to morphed UoMSGFS only. Replace original labels with sequential integers    
   
    # </editor-fold>
#    do_cross_validation = False # Enable/Disable cross validation    
#    fixed_testset = True and (not do_cross_validation) # Whether to train and test using a fixed dataset split
    # <editor-fold desc="[+] UoM_SGFS_Dataset">
    
    # pretrain=True
  
    do_cross_validation = False # Enable/Disable cross validation    
    fixed_testset = True and (not do_cross_validation) # Whether to train and test using a fixed dataset split
    network_fname="vgg_tri_2"
#*******************************
    testdata="UoM"
    bm_alone=False  # Set bm_alone as True if only BM is used
    bm_pi=False  # Set bm_pi as True if only BM & PI is used
    bm_pi_mis=False  # Set bm_pi_mis as True if BM, PI and MIS is used
    # <editor-fold desc="[+] Dataset related configs ...">
    if (not do_cross_validation) and fixed_testset: #For Fixed training and test set
        # train_meta_file = os.path.join(dataset_dir, 'GT','Norm_Sorted_Set_train.txt') # UoM-SGFS dataset
        # train_meta_file = os.path.join(dataset_dir, 'GT','sample_train_rm.txt') #Sample Re-Morphed UoM-SGFS dataset
        train_meta_file = os.path.join(dataset_dir, 'GT','Norm_Sorted_Set_train_rm.txt')# Re-Morphed UoM-SGFS dataset For Grad-CAM training using triplet gain
        # train_meta_file = os.path.join(dataset_dir, 'GT','Norm_Sorted_Set_test_rm.txt')# For Grad-CAM training using triplet loss
        # train_meta_file = os.path.join(root_dir, 'GT','VM', 'Norm_Sorted_VM_Set_Rate_train.txt')  # VM dataset
        
        if (testdata == "UoM"): 
            test_meta_file = os.path.join(dataset_dir, 'GT', 'Norm_Sorted_Set_test.txt') # UoM-SGFS test dataset
            # test_meta_file = os.path.join(dataset_dir, 'GT', 'Norm_Sorted_Set_test1.txt') # Half of the test set
            # test_meta_file = os.path.join(dataset_dir, 'GT', 'Norm_Sorted_Set_test_Vgg16_GT_as_Rank1.txt') # Rank1 as GT
            # test_meta_file = os.path.join(dataset_dir, 'GT','Hiding_game','vgg16','Grad-CAM_Mean_with_blur_txt_files','Norm_Sorted_Set_test_Vgg16_GT_as_Rank1_70percent_hidden.txt') # Rank1 as GT
            # test_meta_file = os.path.join(dataset_dir, 'GT', 'sample_test.txt') #Sample test dataset
        elif (testdata == "VM"):
            test_meta_file = os.path.join(root_dir, 'GT','VM','Norm_Sorted_VM_Set_Rate_test_BM.txt')
            # test_meta_file = os.path.join(root_dir, 'GT','VM','Norm_Sorted_Real_test.txt')   # 7 Real EFIT
            # test_meta_file = os.path.join(root_dir, 'GT','VM','Norm_Sorted_EFIT_Set_test.txt') #44 Success EFITs
    else:  # For cross-validation
        train_meta_file = os.path.join(root_dir, 'GT', 'Norm_Sorted_Set_rm.txt')            
        test_meta_file = os.path.join(root_dir, 'GT', 'Norm_Sorted_Set.txt')
    
    # eg_meta_file = os.path.join(dataset_dir, 'GT','Norm_EG.txt')   # UoM-SGFS Extended Gallery
    # eg_meta_file = os.path.join(dataset_dir, 'GT','Hiding_game','Grad-CAM_Mean_with_blur_txt_files','Norm_EG_UoM_1871_70percent_hidden.txt') 
    eg_meta_file = os.path.join(dataset_dir, 'GT','Norm_EG_UoM_1871.txt') 
    # eg_meta_file = os.path.join(root_dir, 'GT','VM', 'Norm_VM_EG.txt')   # VM Extended Gallery
    # # eg_meta_file = os.path.join(root_dir, 'GT','VM', 'Norm_VM_EG_Real.txt') # 7 Real EFIT EG
    # # eg_meta_file = os.path.join(root_dir, 'GT','VM', 'Norm_VM_EG_Real2.txt') # 44 Success EFITs EG
    
    
    mismatch_cons_check_meta_file = os.path.join(root_dir, 'GT','VM', 'Mismatch_for_lcnn9_model_testing.txt') # Collected human ratings for 200 identities to check the order consistency between the algorithm and humans
    # mismatch_cons_check_meta_file = os.path.join(root_dir, 'GT','VM', 'Norm_Neg_Pairs_Rate.txt') # Collected human ratings for 200 identities to check the order consistency between the algorithm and humans
      
    save_results_filepath=os.path.join(save_results_dir, config_fname_+'.xlsx')
    save_valid_results_filepath=os.path.join(save_results_dir, config_fname_ +'_val.xlsx')
    
    #*******************************
    # For UoM-SGFS dataset    
    load_setB = False
    reset_label = True # Applicable to morphed UoMSGFS and VM dataset only. Replace original labels with sequential integers
    sample_mask = [] # Indicates which samples of each identity from Morphed UoMSGFS are included for training and validation
                    # Set it to None if you want to include all morphs
#     101 samples
    i = 1
    for a_val in [-10, -25, 0, 10, 25]:
        for g_val in range(-4,5,2):
            for w_val in range(-40,41,20):
                for h_val in range(-40,41,20):
                    if a_val in [-10,0,10] and g_val in [-2,0,2] and w_val in [-20,0,20] and h_val in [-20,0,20]:
                        sample_mask.append(i)
                    i += 1
    sample_mask = sample_mask + list(range(i,i+20)) # Include all 20 samples of local variations

### 29 samples 
##    i = 1
##    for a_val in [-10, -25, 0, 10, 25]:
##        for g_val in range(-4,5,2):
##            for w_val in range(-40,41,20):
##                for h_val in range(-40,41,20):
##                    if a_val in [-10,0,10] and g_val in [-2,0,2] and w_val in [0] and h_val in [0]:
##                        sample_mask.append(i)
##                    i += 1
##    sample_mask = sample_mask + list(range(i,i+20)) # Include all 20 samples of local variations  
##    sample_mask = None
#    # </editor-fold>
    
    # optimization options
    # lr_scheduler_type = 'StepLR'
#    lr_scheduler_type = 'CosineAnnealingLR'
#    lr_scheduler_type = 'CosineAnnealingWarmRestarts'
    lr_scheduler_type = 'ReduceLROnPlateau' 
    T_0 = 20
    T_mult = 1
    T_max = 20
    eta_min = 1e-6
    # lr = 1e-4 # learning rate
    lr = 1e-5 # learning rate
    lr_step_size=1 # step size to decay learning rate
    lr_decay_rate=0.95 # decay rate
    weight_decay = 0
#    weight_decay = 1e-4
    n_epochs = 30 # Number of training epochs
    start_epoch = 0 # From which epoch to resume training which is set by Solver._resume()
    
#    checkpoint_file = trained_model_dir /  'vgg_tri_2_Mon_06Jul2020_173546_epoch30.pth' # Hardest triplets for visualization
    checkpoint_file = project_dir /'log_results' /'log_results_Mon_11Oct2021_1043_Malta_SetA_anchor_sketch_type'/'ckt'/'vgg_tri_2_Mon_11Oct2021_144554_epoch30.pth'
    checkpoint_epoch = 10 # Checkpoints will be saved every checkpoint_epoch during training
    if (bm_alone == False and bm_pi == False and bm_pi_mis== False):
        n_bclass = 16 # Number of classes/identities in one mini-batch  
        n_sample_per_cls = 2 # Number of samples per class/identity in one mini-batch
    
    if (bm_alone==True):
        n_bclass = 32 # Number of classes/identities in one mini-batch  # For VM dataset with BM alone  
        n_sample_per_cls = 2 # Number of samples per class/identity in one mini-batch # For VM dataset with BM alone
    elif (bm_pi==True):
        n_bclass = 12 # Number of classes/identities in one mini-batch  # For VM dataset with PI or MIS   
        n_sample_per_cls = 6 # Number of samples per class/identity in one mini-batch  #For VM dataset with PI or MIS 
    elif (bm_pi_mis==True):
        n_bclass = 7 # Number of classes/identities in one mini-batch   # For VM dataset with BM, PI and MIS  
        n_sample_per_cls = 10 # Number of samples per class/identity in one mini-batch # For VM dataset with BM, PI and MIS   
       
    n_worker = 0 # Do not modify
    pin_memory = False # Do not modify
    
    # Paths to trained models
    vgg_tri_2_pth = None
    # vgg_tri_2_pth = project_dir /'log_results' /'log_results_Mon_27Sep2021_1737_Malta_SetA_anchor_sketch_type'/'ckt'/'vgg_tri_2_Mon_27Sep2021_214648_epoch30.pth'
    data_mix=False
    
    n_class = 600 # number of classes/identities in the UoMSGFS dataset
    # n_class = 3227 # number of classes/identities in the VM dataset
    test_set_size = 0.25 # Proportion of the dataset for testing
    
    valid_set_size = 0.2 # Proportion of the training data for validation

    pair_selector = 'all' # 'all' or 'hard'. Applicable to losses involving pairs of embeddings such as OSP and InfoNCELoss
    alt_opt = False # Enable/disable alternative optimization for osp_wass_tn and tri_ce_tn
    train_basenet = False # Whether to fine-tune the base net (all feature extraction layers)
    transfer_learning_opt = 2 # 1 - finetune all layers
                              # 2 - finetune fully connected layers and part of the feature extraction layers
                              # 3 - finetune fully connected layers
    n_fold = 4 # Number of folds for cross validation
    test_ranks = [1, 5, 10, 50, 100]
    percent_ranks=[0.25, 0.5, 1, 2, 5]
    weights=[0.50,0.30,0.15,0.04, 0.01]
    row=0
    col=0
    emd_dist_metric = 'cosine'
    
    tri_loss_param = {'margin':0.1, 'metric': 'euclidean'} # for the triplet loss
    triplet_selector = 'hardest' # 'all', 'random', 'semihard' or 'hardest'. Applicable to the triplet loss

    occm_loss_param = {'lambda_1': 0.1, 'lambda_2': 0.1, 'lambda_3': 0.01} # weights for the loss associated with OSP
    cdl_loss_param = {'w_cel': 1, 'w_tnl': 0.01, 'w_ol': 0.01, 'w_tl': 1} # weights for the tri_ce_tn loss
    ice_loss_param = {'top_k': 32, 'tau': 1, 'metric': 'cosine', 'sort': True} # for the InfoNCELoss
    ce_tri_loss_param = {'w_cel': 1, 'w_tl': 1}
    agt_loss_param = {'margin': 0.1, 'metric': 'euclidean', 'lambda_1': 0.07, 'lambda_2': 0.001}
    osp_wass_tn_loss_param = {'w_cel': 1, 'w_wassl': 1, 'w_ospl': 0.01, 'w_tnl': 0.001, 'w_ol': 0.0}
    
    # For setting up data transform for different net architectures
    vgg_face_dag_meta = {'mean': [129.186279296875, 104.76238250732422, 93.59396362304688], 
                         'std': [1, 1, 1], 
                         'imageSize': [224, 224, 3],
                         'multiplier': 255.0 }
    vgg_data_transform = {'img_resize': 256, 'crop_type': 0, # 0: no crop, 1: centre_crop, 2: random_crop
                          'random_flip': False,
                          'override_meta_imsize': False,
                          'to_grayscale': False}
#    lcnn9_meta = {'mean': [0],
#                 'std': [1],
#                 'imageSize': [128, 128, 3],
#                 'multiplier': 1.0}
#    lcnn9_data_transform = {'img_resize': 144, 'crop_type': 0, 
#                           'random_flip': False,
#                           'override_meta_imsize': False,
#                           'to_grayscale': True}
#    senet_meta = {'mean': [131.0912, 103.8827, 91.4953],
#                 'std': [1, 1, 1],
#                 'imageSize': [224, 224, 3],
#                 'multiplier': 255.0}
#    senet_data_transform = {'img_resize': 256, 'crop_type': 1, 
#                           'random_flip': False,
#                           'override_meta_imsize': False,
#                           'to_grayscale': False}        

    # Set data transform
#    meta = lcnn9_meta
#    data_transform =  lcnn9_data_transform
    meta = vgg_face_dag_meta
    data_transform =  vgg_data_transform
    
    normalize_embedding= True
    existing_model = True   # Set True, if predict_automatic_rating is False and vice-versa
    predict_automatic_rating = False
    filter_low_rating = True
    rating_threshold = 3.5
    consider_ratings_during_train = False  # triplet margin = 0.1
    # consider_ratings_during_train = True   # triplet margin = 1
    # anchor_type='both'
    anchor_type='sketch_only'
##############################################################################
# Global variables used by the Solver class
##############################################################################
import torch.optim as optim
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from dataset import MVMLoader, SMVMLoader, set_config as set_config_dataset

from dataset import MUoMSGFSLoader, MUoMSGFSLoader_CV, set_config as set_config_dataset  # UoM-SGFS dataset
from metrics import MeanLoss, AverageNonzeroTripletsMetric
#from lcnn9_tri_pretrained import get_model, set_config as set_config_net
#from vgg_tri_pretrained import get_model, set_config as set_config_net  # For pretrained model
from vgg_tri_2 import get_model, set_config as set_config_net
# from vgg_tri_2_vis import get_model, set_config as set_config_net # For visualization via Grad-CAM
from triplet_loss import OnlineTripletLoss,OnlineTripletGain_vis,set_config as set_config_loss
from triplet_loss import RandomNegativeTripletSelector, AllNegativeTripletSelector
from triplet_loss import SemihardNegativeTripletSelector, HardestNegativeTripletSelector
from lr_scheduler_wrapper import LRSchedulerWrapper, set_config as set_config_scheduler
from utils import set_config as set_config_utils
import numpy as np

np.random.seed(0)

C = Configuration
set_config_dataset(C)
set_config_net(C)
set_config_loss(C)
set_config_scheduler(C)
set_config_utils(C)
trained_model_dir = C.trained_model_dir
# <editor-fold desc="[+] Initialize DataLoader...">
data_loader = None
if C.do_cross_validation:
    data_loader = MUoMSGFSLoader_CV(C.n_fold)
else:
    data_loader = MUoMSGFSLoader()
# if C.do_cross_validation:
#     data_loader = MVMLoader_CV(C.n_fold)
# else:
#     data_loader = MVMLoader() 
#    data_loader= SMVMLoader()
# </editor-fold>

# <editor-fold desc="[+] Initialize Network ...">
model = get_model()
model.to(C.device)
#summary(model, input_size=(3, 224, 224))  # vgg model
#summary(model, input_size=(1, 128, 128)) # lcnn9 model
# </editor-fold>

# <editor-fold desc="[+] Initialize Criterion ...">
triplet_selectors = {'random': RandomNegativeTripletSelector(),
                     'semihard': SemihardNegativeTripletSelector(),
                     'hardest': HardestNegativeTripletSelector(),
                     'all': AllNegativeTripletSelector()}
if C.visualize:
    # criterion = OnlineTripletLoss_vis(triplet_selectors[C.triplet_selector])
    criterion = OnlineTripletGain_vis(triplet_selectors[C.triplet_selector])
else:
    criterion = OnlineTripletLoss(triplet_selectors[C.triplet_selector])
# </editor-fold>

# <editor-fold desc="[+] Initialize Optimizer ...">
def get_optimizer(model):
    optimizer = optim.Adam(model.parameters(), lr=C.lr, weight_decay=C.weight_decay)
    return optimizer

def get_scheduler(optimizer):
    scheduler = LRSchedulerWrapper(optimizer, C.lr_scheduler_type, last_epoch=-1)
    return scheduler
# </editor-fold>

# <editor-fold desc="[+] Initialize Metrics Recorder ...">
train_loss_recorder = MeanLoss()
valid_loss_recorder = MeanLoss()
train_metric_recorder = AverageNonzeroTripletsMetric()
valid_metric_recorder = AverageNonzeroTripletsMetric()
# </editor-fold>

