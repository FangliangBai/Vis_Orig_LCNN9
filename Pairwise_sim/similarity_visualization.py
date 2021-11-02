# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 11:53:12 2020

@author: Sivapriyaa

Visualization based on the paper "Visualizing Deep Similarity Networks"- Stylianou et.al(2019)
"""

# Visualize pairwise matches

import torch

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
from torchsummary import summary   
from lcnn9_tri_pretrained import get_model,set_config as set_config_net
from lcnn9_tri import get_model,set_config as set_config_net
from similarity_ops import compute_spatial_similarity
from image_ops import combine_image_and_heatmap, combine_horz

import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import cv2
import numpy as np

class Configuration(object):
    project_dir = Path(__file__).resolve().parents[0]  # parents[0] for current working directory
# <editor-fold desc="[+] Pre-trained Model">
    # vgg_face_dag_pth = project_dir / 'pretrained_models' / 'vgg_face_dag.pth'  # weights of the pre-trained model
    # vgg_tri_2_pth = project_dir /'log_results' /'Final_log_results_Mon_11Oct2021_1755_Malta_SetA_anchor_sketch_type'/'ckt'/'vgg_tri_2_Mon_11Oct2021_221223_epoch30.pth'
    #*******************************
    # For our own trained model
    lcnn9_pth =project_dir / 'pretrained_models' /'LightCNN_9Layers_checkpoint.pth.tar' 
    lcnn9_tri_pth=project_dir/ 'log_results'/'log_results_Wed_06Oct2021_1535_Malta_SetA_anchor_sketch_type'/'ckt'/'lcnn9_tri_Wed_06Oct2021_173415_epoch30.pth'
    # lcnn9_tri_pth=project_dir/ 'log_results'/'log_results_Wed_06Oct2021_1535_Malta_SetA_anchor_sketch_type'/'ckt'/'lcnn9_tri_Wed_06Oct2021_173415_epoch30.pth'
    transfer_learning_opt = 1    
    normalize_embedding=True
    #*******************************
    # # For pretrained model
    # lcnn9_pth =project_dir / 'pretrained_models' /'LightCNN_9Layers_checkpoint.pth.tar' 
    # lcnn9_tri_pth=None
    #*******************************
# </editor-fold>
          
def compose_transforms(meta, resize, to_grayscale, crop_type, override_meta_imsize, random_flip):
    """Compose preprocessing transforms for VGGFace model

    The imported models use a range of different preprocessing options,
    depending on how they were originally trained. Models trained in MatConvNet
    typically require input images that have been scaled to [0,255], rather
    than the [0,1] range favoured by PyTorch.

    Args:
        meta (dict): model preprocessing requirements
        resize (int) [256]: resize the input image to this size
        center_crop (bool) [True]: whether to center crop the image
        override_meta_imsize (bool) [False]: if true, use the value of `resize`
            to select the image input size, rather than the properties contained
            in meta (this option only applies when center cropping is not used.

    Return:
        (transforms.Compose): Composition of preprocessing transforms
    """
    normalize = transforms.Normalize(mean=meta['mean'], std=meta['std'])
    im_size = meta['imageSize']
    assert im_size[0] == im_size[1], 'expected square image size'
    if crop_type == 1:
        transform_list = [transforms.Resize(resize),
                          transforms.CenterCrop(size=(im_size[0], im_size[1]))]
    elif crop_type == 2:
        transform_list = [transforms.Resize(resize),
                          transforms.RandomCrop(size=(im_size[0], im_size[1]))]        
    else:
        if override_meta_imsize:
            im_size = (resize, resize)
        transform_list = [transforms.Resize(size=(im_size[0], im_size[1]))]
    if to_grayscale:
        transform_list += [transforms.Grayscale()]
    if random_flip:
        transform_list += [transforms.RandomHorizontalFlip()]
    transform_list += [transforms.ToTensor()]
    transform_list += [lambda x: x * meta['multiplier']]
    transform_list.append(normalize)   
    return transforms.Compose(transform_list)
    

#*****************************************************************************
# vgg_face_dag_meta = {'mean': [129.186279296875, 104.76238250732422, 93.59396362304688], 
#                                          'std': [1, 1, 1], 
#                                          'imageSize': [224, 224, 3],
#                                          'multiplier': 255.0 }
# vgg_data_transform = {'img_resize': 256, 'crop_type': 0, # 0: no crop, 1: centre_crop, 2: random_crop
#                                           'random_flip': False,
#                                           'override_meta_imsize': False,
#                                            'to_grayscale': False}
# meta = vgg_face_dag_meta
# data_transform =  vgg_data_transform

lcnn9_meta = {'mean': [0],
                'std': [1],
                'imageSize': [128, 128, 3],
                'multiplier': 1.0}
lcnn9_data_transform = {'img_resize': 144, 'crop_type': 0, 
                          'random_flip': False,
                          'override_meta_imsize': False,
                          'to_grayscale': True}
meta = lcnn9_meta
data_transform = lcnn9_data_transform
                
def _get_data_transforms():
    return compose_transforms(meta, resize=data_transform['img_resize'], 
                                                  to_grayscale=data_transform['to_grayscale'],
                                                  crop_type=data_transform['crop_type'],
                                                  override_meta_imsize=data_transform['override_meta_imsize'],
                                                  random_flip=data_transform['random_flip'])
                        
C = Configuration
set_config_net(C)

# network="vgg16"
network="lcnn9"

#Read two images
#query_fn="D:/E2ID/Composite2Photo/GT/VM/Norm_VM_Set/ARM_PI_2_am0081.jpg"
#gt_fn="D:/E2ID/Composite2Photo/GT/VM/Norm_VM_Photos/am0081.jpg"

## Read image and do preprocessing
#a = _get_data_transforms()
#
#qimg = a(Image.open(query_fn))   # Open the image using PILLOW
#qimg=preprocess_im(query_fn,mean_im_path) 
#fmimg = _get_data_transforms()(Image.open(gt_fn))
#
## Initialize Network
#model = get_model()
#cuda = torch.cuda.is_available()
#device = torch.device("cuda:0" if cuda else "cpu")
#model=model.to(device)
#summary(model, input_size=(3, 224, 224))  # vgg model
##                summary(model, input_size=(1, 128, 128)) # lcnn9 model
#   
#qimg = qimg.unsqueeze(0)  # Expand the dimension
#fmimg = fmimg.unsqueeze(0) 
#
#qimg_t = qimg.to(device)  # Move to GPU
#fmimg_t = fmimg.to(device)
#
#model.eval()
#emd_query = model(qimg_t)
#first_match = model(fmimg_t)
#
##Take the generated embedding from the model, which is in the form of tuple(emd, target)
#emd_query = emd_query[0] 
#first_match = first_match[0]
#
## Convert into CPU numpy array 
#emd_query = emd_query.detach().cpu().numpy()
#first_match = first_match.detach().cpu().numpy()
#        
## Compute the spatial similarity maps (returns a heatmap that's the size of the last conv layer)
##                emd_query=emd_query.reshape(emd_query.shape[2],emd_query.shape[3], emd_query.shape[1])
##                first_match=first_match.reshape(first_match.shape[2],first_match.shape[3], first_match.shape[1])
##                heatmap1, heatmap2 = compute_spatial_similarity_map(emd_query,first_match)
#heatmap1, heatmap2 = compute_spatial_similarity(emd_query.reshape(-1,emd_query.shape[1]),first_match.reshape(-1,first_match.shape[1]))
#
## Combine the images with the (interpolated) similarity heatmaps.
#im1_with_similarity = combine_image_and_heatmap(load_and_resize(query_fn),heatmap1)
#im2_with_similarity = combine_image_and_heatmap(load_and_resize(gt_fn),heatmap2)
#   
## Merge the two images into a single image and save it out
#combined_image = pil_bgr_to_rgb(combine_horz([im1_with_similarity,im2_with_similarity]))
##                save_fn="sim_heatmap_images/"+network+"/"+fid2+".jpg"
#combined_image.save("XXXX.jpg")

#To produce similarity visualization (heat map) for all the composite-GT pairs in a meta data file
# train_pth="D:/E2ID/Composite2Photo/GT/VM/Norm_Sorted_VM_Set_train.txt"
# train_pth="D:/E2ID/Composite2Photo/GT/VM/sample_train.txt"
train_pth="D:/E2ID/Composite2Photo/GT/VM/test_BM.txt"

i=0
with open(train_pth, "r") as f1:
    for line1 in f1:
        i+=1
        if ("_BM_" in line1):
            query_fn=line1
            query_fn=query_fn.split("@@@",3)
            fid1=query_fn[1]
            query_fn=query_fn[0]
        # if (i % 10 == 0):
        if (i % 2 == 0):
            gt_fn=line1
            gt_fn=gt_fn.split("@@@",3)
            fid2=gt_fn[1]
            gt_fn=gt_fn[0]
            if (fid1==fid2):
                print("The file id is:", fid2)
                print("Query_fn", query_fn)        
                print("Gt_fn", gt_fn)        
                print("****")
                          
                # Read image and do preprocessing
                a = _get_data_transforms()
                qimg = a(Image.open(query_fn))   # Open the image using PILLOW
                
#                fmimg = _get_data_transforms()(Image.open(gt_fn))
                fmimg = a(Image.open(gt_fn)) 
                # Initialize Network
                model = get_model()
                cuda = torch.cuda.is_available()
                device = torch.device("cuda:0" if cuda else "cpu")
                model=model.to(device)
                # summary(model, input_size=(3, 224, 224))  # vgg model
                summary(model, input_size=(1, 128, 128)) # lcnn9 model
                model.eval()
             
                
                query_img=qimg.cpu().numpy()
                gt_img=fmimg.cpu().numpy()
               
                qimg = qimg.unsqueeze(0)  # Expand the dimension
                fmimg = fmimg.unsqueeze(0) 
                
                qimg_t = qimg.to(device)  # Move to GPU
                fmimg_t = fmimg.to(device)
                
                emd_query = model(qimg_t)
                first_match = model(fmimg_t)
                
#                #Take the generated embedding from the model, which is in the form of tuple(emd, target)
#                emd_query = emd_query[0] 
#                first_match = first_match[0]
                
                #************************************************
                # Our own trained model
                # Convert into CPU numpy array 
                emd_query = emd_query[0].detach().cpu().numpy()
                first_match = first_match[0].detach().cpu().numpy()
                #************************************************
                # # Pre-trained model
                # # Convert into CPU numpy array 
                # emd_query = emd_query.detach().cpu().numpy()
                # first_match = first_match.detach().cpu().numpy()
               #************************************************                
                emd_query = np.transpose(emd_query,[0,2,3,1]) # Permute the dimensions of an array
                first_match = np.transpose(first_match,[0,2,3,1])

                e1 = emd_query.reshape(-1,emd_query.shape[-1])
                e2 = first_match.reshape(-1,first_match.shape[-1])
                                              
                # Compute the spatial similarity maps (returns a heatmap that's the size of the last conv layer)
#                e1 = np.reshape(emd_query, [-1,emd_query.shape[1]])  # produces wrong results
#                e2 = np.reshape(first_match, [-1,first_match.shape[1]])
                heatmap1, heatmap2 = compute_spatial_similarity(e1,e2)
                
                # Combine the images with the (interpolated) similarity heatmaps.
                im1_with_similarity = combine_image_and_heatmap(query_img,heatmap1) # (128, 128, 3)
                im2_with_similarity = combine_image_and_heatmap(gt_img,heatmap2) # (128, 128, 3)
               
                # Merge the two images into a single image and save it out
                combined_image = combine_horz([im1_with_similarity,im2_with_similarity])
                save_fn="Results/"+fid2+".jpg"
                combined_image.save(save_fn)
                # break

