# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 16:39:15 2021

@author: Sivapriyaa
"""
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import os
from typing import List, Tuple
project_dir = Path(__file__).resolve().parents[0]  # parents[0] for current working directory

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
    # *************************
    # If not Random replacement # Uncomment the following       
    # Sort CAM pixels in terms of pixel values in ascending order
    ind_arr = np.argsort(cam.flatten()) # Gives the indices of the sorted array of the attention map
    #***********************
    # # # # # Random Replacement
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
        

if __name__ == '__main__':
       img=Image.open("image.bmp")
       query_img=np.array(img)
       
       cam=np.random.rand(7,7)
       percnt_list = [20]
       masked_imgs, attn_masks = gen_CAM_masked_images(query_img, cam, percnt_list, kernel_size=7) 
       for percnt, masked_img, attn_mask in zip(percnt_list, masked_imgs, attn_masks):
                if len(masked_img.shape) > 2:
                    masked_img = masked_img[:, :, ::-1]
                
                final_mask_img=Image.fromarray(masked_img)
                f_mask_name="Masked"+"_"+ str(percnt)+"_percnt.png"
                f_dest1=os.path.join(project_dir,f_mask_name)
                final_mask_img.save(f_dest1)
