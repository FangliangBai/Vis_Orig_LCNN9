# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 14:57:10 2020

@author: Sivapriyaa
"""
import torch
import cv2
import os
import numpy as np
import pickle
from pathlib import Path
from config_2_vgg_tri import C, data_loader, model
from dataset import compose_transforms
from utils import get_embeddings, getFileId, getFileId_VM, calculate_accuracy,\
                             EmbeddingComparator_L2, EmbeddingComparator_Cosine
from kmeans_pytorch import kmeans_predict
from PIL import Image
import matplotlib 
matplotlib.use('TkAgg')
from image_ops import combine_image_and_heatmap
from typing import List, Tuple
import xlsxwriter
project_dir = Path(__file__).resolve().parents[0]  # parents[0] for current working directory
save_dir = os.path.join (project_dir, 'log_results' , 'Triplet_gain_log_results_Tue_12Oct2021_1118_Malta_train_for_Grad-CAM_training', 'ckt')
save_results_dir = os.path.join (project_dir, 'log_results' , 'Triplet_gain_log_results_Tue_12Oct2021_1118_Malta_train_for_Grad-CAM_training', 'ckt')
# save_dir = os.path.join (project_dir, 'log_results' , 'Triplet_loss_log_results_Tue_12Oct2021_1553_Malta_train_for_Grad-CAM_training', 'ckt')
# save_results_dir=save_dir = os.path.join (project_dir, 'log_results' , 'Triplet_loss_log_results_Tue_12Oct2021_1553_Malta_train_for_Grad-CAM_training', 'ckt')
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
    
def _get_data_transforms():
    return compose_transforms(C.meta, resize=C.data_transform['img_resize'], 
                                                  to_grayscale=C.data_transform['to_grayscale'],
                                                  crop_type=C.data_transform['crop_type'],
                                                  override_meta_imsize=C.data_transform['override_meta_imsize'],
                                                  random_flip=C.data_transform['random_flip'])

def predict_faster_gradcam(ori_imgs, test_embeddings,last_conv_fmap , model, kmeans1, channel_weight, channel_address): #query test_data,model_embed, kmeans1, channel_weight, channel_address
    result_faster=[]
    cluster_no_list=[]
    result_cams = []
    # set device
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
        
    test_embeddings =torch.from_numpy(test_embeddings).float().to(device) # Convert numpy array to tensor for test sketches alone
    
    cluster_no=kmeans_predict(test_embeddings,kmeans1,'euclidean', device=device) 
    cluster_no_list=cluster_no.tolist() # Convert tensor into list
    
    idx=0   
    for cno in cluster_no_list:  # for test sketches alone
        last_conv_fmap_=last_conv_fmap[idx]
        last_conv_fmap_=np.moveaxis(last_conv_fmap_,0,-1) #np.moveaxis(a, source,destination)

        cam = np.dot(last_conv_fmap_[:,:,channel_address[cno][0]], channel_weight[cno][0])
        result_cams.append(cam)
        jetcam=combine_image_and_heatmap(ori_imgs[idx],cam)  # BGR image
        result_faster.append(jetcam)  

        idx+=1
       
    return result_faster, result_cams
   
    
def gen_CAM_masked_images(img: np.ndarray, cam: np.ndarray, #percnt_array: List=[20, 50, 70, 90, 99], 
                   percnt_array: List=[20] ,smoothing: bool=True, kernel_size: int=5, zero_replacement: bool=False) -> Tuple[List[np.ndarray], List[np.ndarray]]:
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
    #*************************
    # If not Random replacement # Uncomment the following       
    # Sort CAM pixels in terms of pixel values in ascending order
    ind_arr = np.argsort(cam.flatten()) # Gives the indices of the sorted array of the attention map
    # #***********************
    # # # # Random Replacement
    # import random
    # img_size=img.shape[0]*img.shape[1]    
    # ind_arr=random.sample(range(img_size),img_size)
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

def show_result(model, test_dataset, test_loader, kmeans, channel_weight, channel_address, hidden_percentage):
    original_imgs= []
#    mean_val = [129.186279296875, 104.76238250732422, 93.59396362304688] # mean of vgg16
    
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
         
    heatmap, raw_cams = predict_faster_gradcam(original_imgs,test_embeddings,last_conv_fmap , model, kmeans, channel_weight, channel_address) # Use the grad-weights of the nearest neighbour to visualize the test image
    all_masks=[]
    for idx in range(len(filenames_query)):
            query_index = idx
            qfile_id = getFileId_VM(filenames_query[query_index])
            query_img=cv2.imread(filenames_query[query_index])
            # For hiding games
            cam = raw_cams[idx]
            # percnt_list = [20, 50, 70, 90, 99]
            percnt_list = [hidden_percentage]
            masked_imgs, attn_masks = gen_CAM_masked_images(query_img, cam, percnt_list, kernel_size=7)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            for percnt, masked_img, attn_mask in zip(percnt_list, masked_imgs, attn_masks):
                if len(masked_img.shape) > 2:
                    # masked_img = masked_img[:, :, ::-1]
                    masked_img1=masked_img[:, :, ::-1] # Convert BGR to RGB
                    
                final_mask_img=Image.fromarray(masked_img1) # Convert numpy array into PILLOW image
                f_mask_name=str(qfile_id)+"_"+ str(percnt)+"_percnt.png"
                f_dest1=os.path.join(save_dir,'HG_masks', f_mask_name)
                final_mask_img.save(f_dest1)
                
                # masked_img=Image.fromarray(masked_img) # Convert numpy array into PILLOW image
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

# def show_result_eg(model, eg_dataset, eg_loader, kmeans, channel_weight, channel_address):
#     original_imgs= []
    
#     emd_eg, last_conv_fmap, _= get_embeddings(model, data_loader.eg_loader)   
#     filenames_test = data_loader.eg_loader.dataset.filenames.tolist()
#     filenames_query = filenames_test
    
#     for ix in range(len(filenames_query)):
#          query_index = ix
#          qfile_id = getFileId_VM(filenames_query[query_index])
#          query_img=cv2.imread(filenames_query[query_index])  # BGR
#          original_imgs.append(query_img)
         
#     heatmap, raw_cams = predict_faster_gradcam(original_imgs,emd_eg,last_conv_fmap , model, kmeans, channel_weight, channel_address) # Use the grad-weights of the nearest neighbour to visualize the test image
#     for idx in range(len(filenames_query)):
#             query_index = idx
#             qfile_id = getFileId_VM(filenames_query[query_index])
#             query_img=cv2.imread(filenames_query[query_index])
#             vis_img=heatmap[idx]
            
#             # # Merge the two images into a single image and save it out
#             # combined_image = combine_horz([query_img,vis_img])
#             # if "_" in qfile_id:
#             #     vis_fname="Test_photo_"+str(qfile_id)+".jpg"
#             # else:
#             #     vis_fname="Test_sketch_"+str(qfile_id)+".jpg"
#             # dir_name1='HG_masks'
#             # dir_name2='Grad-CAM_vgg16_hard_tri_1_malta_test_heatmaps'
#             # # dir_path=os.path.join(save_dir,dir_name1, dir_name2)
#             # dir_path=os.path.join(save_dir,'HG_masks', 'Grad-CAM_vgg16_hard_tri_1_malta_test_heatmaps')
#             #************
#             # if not os.path.exists(dir_path):
#             #     os.mkdir(dir_path)
            
#             # f_dest=os.path.join(dir_path,vis_fname)
#             # if (idx <=4): # Save only first 5 images
#             #     combined_image.save(f_dest)
#             # combined_image.save(f_dest) # Uncomment to save the heatmaps
#             #************
#             # For hidding games
#             save_dir1=os.path.join(save_dir,'HG_masks')
#             cam = raw_cams[idx]
#             # percnt_list = [20, 50, 70, 90, 99]
#             percnt_list = [20]
#             masked_imgs, attn_masks = gen_CAM_masked_images(query_img, cam, percnt_list, kernel_size=7)
#             if not os.path.exists(C.vis_dir):
#                 os.makedirs(C.vis_dir, exist_ok=True)
#             for percnt, masked_img, attn_mask in zip(percnt_list, masked_imgs, attn_masks):
#                 if len(masked_img) > 2:
#                     masked_img = masked_img[:, :, ::-1]
                
#                 final_mask_img=Image.fromarray(masked_img)
#                 f_mask_name=str(qfile_id)+"_"+ str(percnt)+"_percnt.png"
#                 f_dest1=os.path.join(save_dir1,f_mask_name)
#                 final_mask_img.save(f_dest1)
#                 # final_mask_img.save(save_dir1 / f"{qfile_id}_{percnt}.png")
#                 # final_mask_img.save(save_dir1 /f_mask_name)
#                 #**********
#                 # Binary mask
#                 # final_attn_mask=Image.fromarray(attn_mask.squeeze())
#                 # f_attn_name=str(qfile_id)+"_percnt_mask.png"
#                 # f_dest2=os.path.join(save_dir1,f_attn_name)
#                 # final_attn_mask.save(f_dest2)
#                 #**********
#                 # final_attn_mask.save(save_dir1 / f"{qfile_id}_{percnt}_mask.png")
#                 # final_attn_mask.save(save_dir1 / f_attn_name)            

if __name__ == '__main__':
    
        hidden_percentage = 20
        
        # load csv 
        print("Loading CSV files & K-means Pickle file...")
        caname=os.path.join(save_dir,'channel_address.csv') # Saved Channel address (aggregated from the grad-weights of the training embeddings)
        cwname=os.path.join(save_dir,'channel_weight.csv')  # Saved Channel weights
        
        with open(caname) as f1:
            n_cols1 = len(f1.readline().split(","))
            channel_address=np.loadtxt(caname, delimiter=",", dtype="int", usecols=np.arange(0,n_cols1-1))
        with open(cwname) as f2:
            n_cols2 = len(f2.readline().split(","))
            channel_weight=np.loadtxt(cwname, delimiter=",", dtype="str", usecols=np.arange(0,n_cols2-1))
        channel_weight=channel_weight.astype(np.float32) # Convert string array into float array
        fname=os.path.join(save_dir,'grad_cam_kmeans.pickle') # Saved cluster centers of size 100352 (14x14x512) in this pickle file 
        with open(fname, "rb") as f:
            kmeans = pickle.load(f)  
            
        all_masks=show_result(model, data_loader.test_dataset, data_loader.test_loader, 
                              kmeans, np.array(channel_weight), np.array(channel_address),
                              hidden_percentage) 
        # To produce masks for Extended Gallery as well
        # show_result_eg(model, data_loader.dataset_eg, data_loader.eg_loader, kmeans, np.array(channel_weight), np.array(channel_address)) 
        wb=xlsxwriter.Workbook(save_results_dir)
        sheet_names=['SheetA','SheetB']
        sheet_nameA=sheet_names[0]    
        sheet_nameB=sheet_names[1]   
        row=0;col=0
        acc, overall_acc, cal_acc = UoM_test(all_masks, wb,sheet_nameA, sheet_nameB, row, col)
        print("Rank-1 accuracy is {0:.2f}%".format(acc[0]*100))