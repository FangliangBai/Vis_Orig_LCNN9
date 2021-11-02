# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 09:33:26 2021

@author: Sivapriyaa
Visualization Technique - Pairwise similarity map
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Tuple
import torch
import os
from config_1_lcnn9_tri import C, data_loader, model
from dataset import compose_transforms
from similarity_ops import compute_spatial_similarity
from image_ops import combine_image_and_heatmap, combine_horz
from utils import get_embeddings, getFileId, getFileId_VM, calculate_accuracy,\
                             EmbeddingComparator_L2, EmbeddingComparator_Cosine
import xlsxwriter
project_dir = Path(__file__).resolve().parents[0]  # parents[0] for current working directory
save_dir=os.path.join(project_dir,'results','HG_masks')
save_results_dir=os.path.join(project_dir,'results')
 # Setting up GPU device
device_id = '0'
cuda = torch.cuda.is_available()
if cuda:
    device = torch.device('cuda:' + device_id)
else:
    device = torch.device('cpu')     
if C.emd_dist_metric == 'euclidean':
    comparator = EmbeddingComparator_L2()
elif C.emd_dist_metric == 'cosine':
    comparator = EmbeddingComparator_Cosine()
    
 # Evaluating the visualization technique via Hiding game   
def gen_CAM_masked_images(img: np.ndarray, cam: np.ndarray, #percnt_array: List=[20, 50, 70, 90, 99], 
                   percnt_array: List=[99] ,smoothing: bool=True, kernel_size: int=5, zero_replacement: bool=False) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generate CAM masked images for the hiding game

    Parameters
    ----------
    img : np.ndarray
        Original image.
    cam : np.ndarray
        CAM image.
    percnt_array : LIST, optional
        Pencentage array of how many pixels to hide from the image. The default is [20, 50, 70, 90, 99].
    smoothing : bool, optional
        If true the attention masks will be blurred by Gaussian kernels of which the size is kernel_size. The default is True.
    kernel_size : int, optional
        The size of Gaussian kernel for smoothing
    zero_replacement : bool, optional
        If true, the masked pixels will be set to zero or the mean value of the image otherwise.
        The default is False.

    Returns
    -------
    A list of masked images and a list of attention masks

    """
    if img.dtype != np.uint8:
        raise TypeError("The input image must be int8 type")
    masked_imgs = []
    attn_masks = []
    img_h, img_w = img.shape[0], img.shape[1]
    
    minval = np.min(cam)
    cam = (cam - minval) / (np.max(cam) - minval) * 255 # Normalize
    cam = cam.astype(np.uint8)
    cam = Image.fromarray(cam)
    cam = cam.resize((img_w, img_h), Image.ANTIALIAS)    # Added    
    cam = np.array(cam, dtype=np.uint8)
    # *************************
    # If not Random replacement # Uncomment the following       
    # Sort CAM pixels in terms of pixel values in ascending order
    # ind_arr = np.argsort(cam.flatten()) # Gives the indices of the sorted array of the attention map
    #***********************
    # # # # # Random Replacement
    import random
    img_size=img.shape[0]*img.shape[1]    
    ind_arr=random.sample(range(img_size),img_size)
    # # #***********************
    empty_img = np.zeros((img_h, img_w), dtype=np.float32)
    if not zero_replacement:
        meanval = np.mean(img.flatten()) # Mean replacement # default
        empty_img = np.ones((img_h, img_w), dtype=np.float32) * meanval
    if len(img.shape) > 2:
        empty_img = empty_img[:, :, np.newaxis]
    
    for percnt in percnt_array:
        n_hid_pixel = int(0.01 * percnt * img_h * img_w)
        mask = np.ones((img_h * img_w, ), dtype=np.float32)
        mask[ind_arr[:n_hid_pixel]] = 0
        mask = mask.reshape((img_h, img_w))
        if smoothing: # default is True
            mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
        if len(img.shape) > 2:
            mask = mask[:, :, np.newaxis]
        masked_img = img.astype(np.float32) * mask + empty_img * (1 - mask) #Eg: Original image and Empty img(Mean with blur or Zeros) 
        masked_imgs.append(masked_img.astype(np.uint8))
        attn_masks.append((mask * 255).astype(np.uint8))
        
    return masked_imgs, attn_masks
    
               
def _get_data_transforms():
    return compose_transforms(C.meta, resize=C.data_transform['img_resize'], 
                                                  to_grayscale=C.data_transform['to_grayscale'],
                                                  crop_type=C.data_transform['crop_type'],
                                                  override_meta_imsize=C.data_transform['override_meta_imsize'],
                                                  random_flip=C.data_transform['random_flip'])
                        
def compute_sim_map_attention(test_meta_file):
    i=0
    finalcam1=[]
    finalcam2=[]
    heatmap1=[]
    heatmap2=[]
    with open(test_meta_file, "r") as f1:
        for line1 in f1:
            i+=1
            if ("SetA" in line1):
                query_fn=line1
                query_fn=query_fn.split("\n")[0]
                fid1=query_fn.split("@@@",2)[1]
                query_fn=query_fn.split("@@@",2)[0]
            if (i % 3 == 0):
                gt_fn=line1
                gt_fn=gt_fn.split("\n")[0]
                fid2=gt_fn.split("@@@",2)[1]
                gt_fn=gt_fn.split("@@@",2)[0]
                if (fid1==fid2):
                    print("The file id is:", fid2)
                    print("Query_fn", query_fn)        
                    print("Gt_fn", gt_fn)        
                    print("****")
                              
                    # Read image and do preprocessing
                    qimg = _get_data_transforms()(Image.open(query_fn))   # Open the image using PILLOW
                    fmimg = _get_data_transforms()(Image.open(gt_fn))
                    
                    query_img=qimg.cpu().numpy()
                    gt_img=fmimg.cpu().numpy()
                   
                    qimg = qimg.unsqueeze(0)  # Expand the dimension
                    fmimg = fmimg.unsqueeze(0) 
                    
                    qimg_t = qimg.to(device)  # Move to GPU
                    fmimg_t = fmimg.to(device)
                    model.to(device)
                    model.eval()
                    emd_query = model(qimg_t)
                    first_match = model(fmimg_t)
                
                    #************************************************
                    # Our own trained model
                    # Convert into CPU numpy array 
                    emd_query = emd_query[0].detach().cpu().numpy() # last convolutional feature map
                    first_match = first_match[0].detach().cpu().numpy()
                    #************************************************
                    # # # Pre-trained model
                    # # Convert into CPU numpy array 
                    # emd_query = emd_query.detach().cpu().numpy()
                    # first_match = first_match.detach().cpu().numpy()
                   #************************************************      
    
                    emd_query = np.transpose(emd_query,[0,2,3,1]) # Permute the dimensions of an array # (1, 7, 7, 512)
                    first_match = np.transpose(first_match,[0,2,3,1])
    
                    e1 = emd_query.reshape(-1,emd_query.shape[-1])     # e1.shape (49,512)
                    e2 = first_match.reshape(-1,first_match.shape[-1])
                                                  
                    # Compute the spatial similarity maps (returns a heatmap that's the size of the last conv layer)
                    cam1, cam2 = compute_spatial_similarity(e1,e2) # (7,7)
                    
                    # Combine the images with the (interpolated) similarity heatmaps.
                    hmap1 = combine_image_and_heatmap(query_img,cam1) # (224, 224, 3)
                    hmap2 = combine_image_and_heatmap(gt_img,cam2) # (224, 224, 3)
                   
                    # Merge the two images into a single image and save it out
                    combined_image = combine_horz([hmap1,hmap2])
                    save_fn="results/"+fid2+".jpg"
                    combined_image.save(save_fn)
                    # Append CAM and heatmap of both the query img and the ground truth img
                    finalcam1.append(cam1)
                    finalcam2.append(cam2)
                    heatmap1.append(hmap1)
                    heatmap2.append(hmap2)
    return finalcam1, heatmap1
                                    
def show_result(model, test_dataset, test_loader, raw_cams, heatmap):
    original_imgs= []
    
    test_embeddings, last_conv_fmap, labels = get_embeddings(model, test_loader)    # Added for visualization 
    test_embeddings=test_embeddings[::3]
    last_conv_fmap=last_conv_fmap[::3]
    
    test_cls = labels[::3]
    ctoi = data_loader.test_loader.dataset.label_to_indices
    test_indices = np.concatenate([ctoi[c] for c in test_cls])
    filenames_test = data_loader.test_loader.dataset.filenames[test_indices].tolist()
    filenames_query = filenames_test[::3]
    
    for ix in range(len(filenames_query)):
         query_index = ix
         qfile_id = getFileId_VM(filenames_query[query_index])
         query_img=cv2.imread(filenames_query[query_index])  # BGR
         original_imgs.append(query_img)        
    all_masks=[]
    for idx in range(len(filenames_query)):
            query_index = idx
            qfile_id = getFileId_VM(filenames_query[query_index])
            query_img=cv2.imread(filenames_query[query_index])
            # For hiding games
            cam = raw_cams[idx]
            # percnt_list = [20, 50, 70, 90, 99]
            percnt_list = [99]
            masked_imgs, attn_masks = gen_CAM_masked_images(query_img, cam, percnt_list, kernel_size=7) # masked_imgs[0].shape - (256, 256, 3)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            for percnt, masked_img, attn_mask in zip(percnt_list, masked_imgs, attn_masks):
                if len(masked_img.shape) > 2:
                    # masked_img = masked_img[:, :, ::-1]
                    masked_img1=masked_img[:, :, ::-1] 
                    
                final_mask_img=Image.fromarray(masked_img1) # Convert numpy array into PILLOW image
                f_mask_name=str(qfile_id)+"_"+ str(percnt)+"_percnt.png"
                f_dest1=os.path.join(save_dir,f_mask_name)
                final_mask_img.save(f_dest1)
                
                masked_img = _get_data_transforms()(final_mask_img)   # Transform it

                f_masked_img=masked_img.unsqueeze(0).detach().cpu() # Move GPU Tensor to CPU
                all_masks.append(f_masked_img) # Append CPU tensor
                
                
    return all_masks
               
                
def UoM_test(all_masks, wb, sheet_nameA, sheet_nameB, row, col, epoch=None):                  
        # Calculate embeddings of the query images        
        print('\nLoading query images...')   
        embeddings, _, labels = get_embeddings(model, data_loader.test_loader)    
        model.to(device)
        model.eval()
        mask_embeddings=[]
        for idx in range(len(all_masks)):
            data=all_masks[idx]
            target = None
            if type(data) in (tuple, list):
                target = data[1]  # GT File_ID
                data = data[0]
                if type(data) in (tuple,list):
                    data = tuple(d.to(C.device) for d in data) # Convert to cuda tuple  # For 2 trained face parsers
                else:
                    # data = data[0]  # Extract Only data
                    data = data.to(C.device)
            else:
                data = data.to(C.device)
            if target is not None:  
                if type(target) == torch.Tensor:
                    _labels = target.detach().numpy()
                elif type(target) == tuple:
                    _labels = list(target)
                elif type(target)==np.ndarray:
                    _labels=target
                else:
                    raise TypeError("Unexpected data type of target")
            
            mask_embeddings_, feature_map_ = model.get_embedding(data)  # feature_map_ (Added for visualization)
            
            mask_embeddings_=mask_embeddings_.cpu()  # Convert CUDA tensor to CPU tensor
            mask_embeddings.append(mask_embeddings_)
            
        mask_embeddings = np.vstack(mask_embeddings)
        
        if comparator.get_metric_type() == 'cosine':
            embeddings = embeddings / (1e-10 + np.linalg.norm(embeddings, axis=1, keepdims=True))  # This is to make sure it is normalized before cosine calculation
            mask_embeddings = mask_embeddings / (1e-10 + np.linalg.norm(mask_embeddings, axis=1, keepdims=True))  # This is to make sure it is normalized before cosine calculation
        
        emd_setA_query = mask_embeddings 
        # emd_setA_query = embeddings[::3]
        # emd_setB_query = embeddings[1::3]
        emd_gt = embeddings[2::3]   
        query_cls = set(labels.tolist())
        
        # Calculate embeddings of the gallery
        print('\nLoading test gallery...')
        emd_eg, _, _= get_embeddings(model, data_loader.eg_loader)
    
        if comparator.get_metric_type() == 'cosine':
            emd_eg = emd_eg / (1e-10 + np.linalg.norm(emd_eg, axis=1, keepdims=True))
            
        all_cls = [int(getFileId(fn)) for fn in data_loader.eg_loader.dataset.filenames[:C.n_class]]
        cls_ind = {cl:i for i,cl in enumerate(all_cls)}
        nongt_cls = list(set(all_cls) - query_cls)
        nongt_ind = [cls_ind[cl] for cl in nongt_cls]
        nongt_ind = nongt_ind + np.arange(C.n_class, emd_eg.shape[0]).tolist()  
        emd_eg = emd_eg[nongt_ind]
        emd_eg = np.vstack([emd_gt, emd_eg])
        
        print('\nAccuracies for Set A with Extended Gallery:')
        acc_setA, overall_setA, cal_acc_setA = calculate_accuracy(emd_eg, emd_setA_query, C.test_ranks,C.percent_ranks, comparator, wb,sheet_nameA, row, col)
        # print('\nAccuracies for Set B with Extended Gallery:')
        # acc_setB, overall_setB, cal_acc_setB = calculate_accuracy(emd_eg, emd_setB_query, C.test_ranks,C.percent_ranks, comparator,wb,sheet_nameB, row, col)  
        
        # Show some matching results
        import cv2
        from imutils import build_montages
        test_cls = labels[::3]
        ctoi = data_loader.test_loader.dataset.label_to_indices
        test_indices = np.concatenate([ctoi[c] for c in test_cls])
        filenames_test = data_loader.test_loader.dataset.filenames[test_indices].tolist()
        filenames_query = filenames_test[::3]
        filenames_eg = filenames_test[2::3]
        filenames_eg_ = data_loader.eg_loader.dataset.filenames[nongt_ind].tolist()
        filenames_eg += filenames_eg_
        cnt=0
        print("********************")
        print("The matched image files are:");
        for query_index in range(len(filenames_query)):
            qfile_id = getFileId(filenames_query[query_index])
            query_img=cv2.imread(filenames_query[query_index])
            cos_sim, indices = comparator.get_rank_list(emd_setA_query[query_index], 
                                              emd_eg, top_n=1)
            
            efile_id = [getFileId(filenames_eg[index]) for index in indices]    
            images = [cv2.imread(filenames_eg[index]) for index in indices]
            images.append(query_img)
            images = [cv2.resize(image, (200,200)) for image in images]
            # result = build_montages(images, (200, 200), (5,5))[0]
            
            # print("The query image file: {0}".format(filenames_query[query_index]))
            filenames_top = [filenames_eg[i] for i in indices]
            # for fn in filenames_top:
            #     print("{0}".format(fn))
            
            if qfile_id in efile_id:
                for i,fid in enumerate(efile_id):
                    if fid == qfile_id:
                        # print("\nFOUND THE MATCH: {0}".format(filenames_top[i]))
                        # print("FOUND THE MATCH")
                        print(filenames_top[i])
                        cnt+=1
                        break
            # else:
            #     print("NO MATCH FOUND")
        print("The total number of test sketches which matches GT as Top1 match is:", cnt)
        print("********************")
        # cv2.imshow("Matched Results", result)
        # cv2.waitKey(0)
        # cv2.imwrite("example_3.png", result)
        # cv2.destroyAllWindows()  
    
        # return acc_setA, acc_setB, overall_setA, overall_setB, cal_acc_setA, cal_acc_setB
        return acc_setA, overall_setA, cal_acc_setA


if __name__ == '__main__':
      
       raw_cam1, heatmap1 = compute_sim_map_attention(C.test_meta_file) # Returns list of cams (7,7) and heatmaps (224, 224, 3)
       all_masks = show_result(model, data_loader.test_dataset, data_loader.test_loader, raw_cam1 ,heatmap1) # Return the all the masked images
       wb=xlsxwriter.Workbook(save_results_dir)
       sheet_names=['SheetA','SheetB']
       sheet_nameA=sheet_names[0]    
       sheet_nameB=sheet_names[1]   
       row=0;col=0
       acc, overall_acc, cal_acc = UoM_test(all_masks, wb,sheet_nameA, sheet_nameB, row, col)
       print("Rank-1 accuracy is {0:.2f}%".format(acc[0]*100))
       