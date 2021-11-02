import numpy as np
from numpy import matlib as mb # matlib must be imported separately

def compute_spatial_similarity(conv1,conv2):
    """
    Takes in the last convolutional layer from two images, computes the pooled output
    feature, and then generates the spatial similarity map for both images.
    conv1 [H*W, Channel], conv2 [H*W, Channel]
    
    Paper:
        Stylianou, Abby, Richard Souvenir, and Robert Pless.
        "Visualizing deep similarity networks."
        2019 IEEE Winter Conference on Applications of Computer Vision (WACV). IEEE, 2019.

    """    
    pool1 = np.mean(conv1,axis=0) # axis=0 represents rows (Channel, )
    pool2 = np.mean(conv2,axis=0) # (Channel, )
    
    out_sz = (int(np.sqrt(conv1.shape[0])),int(np.sqrt(conv1.shape[0])))  # (H, H)
    # Equ 3 in paper
    
    conv1_normed = conv1 / np.linalg.norm(pool1) / conv1.shape[0]  # (H^2, C) Normalize - np.linalg.norm -> Frobenius norm 
    conv2_normed = conv2 / np.linalg.norm(pool2) / conv2.shape[0]  # (H^2, C)
    
    im_similarity = np.zeros((conv1_normed.shape[0],conv1_normed.shape[0]))  # (H^2, H^2)
    
    for zz in range(conv1_normed.shape[0]):   # loop for each pixel
        repPx = mb.repmat(conv1_normed[zz,:],conv1_normed.shape[0],1) # (H^2, c) - Matlib repmat - Repeat a matrix MXN times
        im_similarity[zz,:] = np.multiply(repPx,conv2_normed).sum(axis=1)
        
    similarity1 = np.reshape(np.sum(im_similarity,axis=1),out_sz)
    similarity2 = np.reshape(np.sum(im_similarity,axis=0),out_sz)
    
    return similarity1, similarity2
