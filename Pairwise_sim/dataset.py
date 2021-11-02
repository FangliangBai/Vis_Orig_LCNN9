# Stratified - Splitting the training set into train and validation set (without overlap) 

import numpy as np
#from skimage import io
from PIL import Image
import os
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler, Sampler
import pandas as pd
import matplotlib.pyplot as plt
import torch
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import KFold
#from config import Configuration as C
import math
from statistics import median
import torchvision.transforms as transforms
from skimage import measure, transform
import cv2
import itertools
from itertools import chain
C = None

def set_config(_C):
    global C
    C = _C

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def warp_image(img, shp_src, shp_dst):
    tform=transform.PiecewiseAffineTransform()
    tform.estimate(shp_dst, shp_src)
    t_img = transform.warp(img, tform)
    return t_img
    
def data_mix(x, target):
    if type(x) in (tuple, list):
        if C.data_mix_type == 'warp':
            x, pts, _ = x
            pts = pts.detach().cpu().numpy()
        else:
            x, _ = x
    if type(target) in (tuple, list):
        target = target[0]
    x = x.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    clas, mode = target[:,0], target[:,1]
    clas_set = np.unique(clas)
    idx = np.arange(len(x))
    ctoi_c = {cl: idx[np.where(np.logical_and(clas==cl, mode==0))].tolist() for cl in clas_set}
    ctoi_p = {cl: idx[np.where(np.logical_and(clas==cl, mode==1))].tolist() for cl in clas_set}
    ctoi = [ctoi_c, ctoi_p]
    clas_set_p = np.random.permutation(clas_set)
    
    x_mix, _clas, _mode, beta = [], [], [], []
    
    for (cl, _cl) in zip(clas_set.tolist(), clas_set_p.tolist()):
        if cl == _cl:
            continue
        _beta = np.random.beta(C.beta, C.beta)
        
        for m, _ctoi in enumerate(ctoi): # loop over both modalities
            idx_a, idx_b = _ctoi[cl], _ctoi[_cl]
            if not idx_a or not idx_b:
                continue
            if len(idx_a) > len(idx_b):
                idx_b = idx_b + np.random.choice(idx_b, len(idx_a)-len(idx_b), replace=True).tolist()
            elif len(idx_a) < len(idx_b):
                idx_a = idx_a + np.random.choice(idx_a, len(idx_b)-len(idx_a), replace=True).tolist()
            if C.data_mix_type == 'alpha_blend':
                _x_mix = x[idx_a] * _beta + x[idx_b] * (1 - _beta)
            elif C.data_mix_type == 'warp':
                _x_mix = []
                for _idx_a, _idx_b in zip(idx_a, idx_b):
                    shp_a, shp_b = np.copy(pts[_idx_a]), np.copy(pts[_idx_b])
                    shp_a /= (shp_a[-1, :][np.newaxis, ...] + 1)
                    shp_b /= (shp_b[-1, :][np.newaxis, ...] + 1)
                    shp_a = np.floor(shp_a * (x.shape[3], x.shape[2]))
                    shp_b = np.floor(shp_b * (x.shape[3], x.shape[2]))                    
                    img_a = np.copy(np.transpose(x[_idx_a], axes=[1,2,0]))
                    img_b = np.copy(np.transpose(x[_idx_b], axes=[1,2,0]))
                    if _beta < 0.2:
                        _img_a = warp_image(img_a, shp_a, shp_b)
                        _img_b = img_b
                    elif _beta > 0.8:
                        _img_a = img_a
                        _img_b = warp_image(img_b, shp_b, shp_a)
                    else:
                        shp_m = np.floor(_beta * shp_a + (1 - _beta) * shp_b)
                        _img_a = warp_image(img_a, shp_a, shp_m)
                        _img_b = warp_image(img_b, shp_b, shp_m)

                    img_mix = _beta * _img_a + (1 - _beta) * _img_b
                    
                    # img_a *= 255
                    # img_b *= 255
                    # _img_a *= 255
                    # _img_b *= 255
                    # for i in range(len(shp_a)):
                    #     cv2.circle(img_a, (int(shp_a[i,0]),int(shp_a[i,1])), 1, (255, 255, 255), 2)
                    # for i in range(len(shp_b)):
                    #     cv2.circle(img_b, (int(shp_b[i,0]),int(shp_b[i,1])), 1, (255, 255, 255), 2)
                    # for i in range(len(shp_m)):
                    #     cv2.circle(_img_a, (int(shp_m[i,0]),int(shp_m[i,1])), 1, (255, 255, 255), 2)
                    #     cv2.circle(_img_b, (int(shp_m[i,0]),int(shp_m[i,1])), 1, (255, 255, 255), 2)
                    # cv2.imwrite('img_a.jpg', img_a)
                    # cv2.imwrite('img_b.jpg', img_b)
                    # cv2.imwrite('_img_a.jpg', _img_a)
                    # cv2.imwrite('_img_b.jpg', _img_b)
                    # cv2.imwrite('img_mix.jpg', img_mix*255)
                    _x_mix.append(np.transpose(img_mix, axes=[2,0,1])[np.newaxis, ...])
                _x_mix = np.vstack(_x_mix)   
            x_mix.append(_x_mix)
            if _beta < 0.3:
                _clas.append(np.array([[_cl, _cl]], dtype=target.dtype).repeat(len(idx_a), axis=0))
                #beta = beta + [1]*len(idx_a)
            elif _beta > 0.7:
                _clas.append(np.array([[cl, cl]], dtype=target.dtype).repeat(len(idx_a), axis=0))
                #beta = beta + [1]*len(idx_a)
            else:
                _clas.append(np.array([[cl, _cl]], dtype=target.dtype).repeat(len(idx_a), axis=0))
            beta = beta + [_beta]*len(idx_a)
            _mode = _mode + [m]*len(idx_a)
        
    x_mix.append(x)
    _clas.append(clas.reshape(-1,1).repeat(2, axis=1))
    _mode = _mode + mode.tolist()
    beta = beta + [1]*len(x)
    
    x = torch.tensor(np.vstack(x_mix), dtype=torch.float32) 
    clas = torch.tensor(np.vstack(_clas), dtype=torch.long)
    mode = torch.tensor(_mode, dtype=torch.long)
    beta = torch.tensor(beta, dtype=torch.float32)
    
    return x, (clas, mode, beta)
    
    
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

# One Sketch will have one GT
class UoMSGFSDataset(Dataset):
    """
    Conventional dataset
    """

    def __init__(self, data_frame):
        self.frame = data_frame
        self.transform = self._get_data_transforms()
        self.loaded_file_idx = -1
        self.filenames = self.frame.iloc[:,0]
        self.labels = None
        self.label_set = None
        self.label_to_indices = None
        # The following initialization is for BalancedBatchSampler or its kind
        if self.frame.shape[1] > 1:
            self.labels = self.frame.iloc[:,1].to_numpy()    # array([2,2,2,...526,526,526]) len(self.labels)=1440
            self.labels=self.labels.tolist()
#            self.label_set=sorted(set(self.labels[:,0]), key=self.labels.index) # 
            self.label_set=sorted(set(self.labels), key=self.labels.index) # 
#            self.label_set = list(set(self.labels))          # [2,3,5,6,13,19.....525,526]len(self.label_set)=480
            self.label_to_indices = {}
            k = 0
            for label in self.label_set:
                indexes = []
                while k < len(self.labels) and self.labels[k] == label:
                    indexes.append(k)
                    k += 1
                self.label_to_indices[label] = indexes           # {   2:[0,1,2], 3:[3,4,5], 5:[6,7,8]....     }
        print("Success")
#            while k < len(self.labels):
#                label = self.labels[k]
#                indexes = []
#                while k < len(self.labels) and self.labels[k] == label:
#                    indexes.append(k)
#                    k += 1
#                self.label_to_indices[label] = indexes
        
    def __getitem__(self, index):
        timg = []
        
        if type(index) in (tuple, list):
            if self.transform is not None:   
                timg = [self.transform(Image.open(self.filenames[i])) for i in index]
                if type(timg[0]) is torch.Tensor:
                    timg = torch.cat([x.unsqueeze(0) for x in timg]) # Expand the dimension
                else:
                    timg = np.concatenate([np.array(x, dtype=np.float32)[np.newaxis,...] for x in timg])
                    timg = np.transpose(timg, axes=[0, 3, 1, 2])
            else:
                timg = [np.array(Image.open(self.filenames[i]), dtype=np.float32) for i in index]
                timg = np.concatenate([x[np.newaxis,...] for x in timg])
                timg = np.transpose(timg, axes=[0, 3, 1, 2])
        else:
            if self.transform is not None:           
                timg = self.transform(Image.open(self.filenames[index]))
                if type(timg) is not torch.Tensor:
                    timg = np.array(timg, dtype=np.float32)
                    timg = np.transpose(timg, axes=[2, 0, 1])
            else:
                timg = np.array(Image.open(self.filenames[index]), dtype=np.float32)
                timg = np.transpose(timg, axes=[2, 0, 1])
           
        label = None
        if self.labels is not None:
            index=int(index)
            label = self.labels[index]
            return timg, label
        else:
            return timg

    def __len__(self):
        return len(self.filenames)
    
    def n_class(self):
        if self.label_set is not None:
            return len(self.label_set)
        else:
            return len(self.filenames)
    
    def get_labels(self):
        return self.labels
    
    def get_label_to_indices(self):
        return self.label_to_indices   
    
    def _get_data_transforms(self):
        return compose_transforms(C.meta, resize=C.data_transform['img_resize'], 
                                  to_grayscale=C.data_transform['to_grayscale'],
                                  crop_type=C.data_transform['crop_type'],
                                  override_meta_imsize=C.data_transform['override_meta_imsize'],
                                  random_flip=C.data_transform['random_flip']) 
    
# Multiple sketches have one GT
class MVMDataset(Dataset):
    """
    VM Dataset (each identity with 9 composites) with three labels: subject IDs, modality and Human Ratings
    """
    def __init__(self, data_frame, order_rating=False):

        """
        reset_label: the purpose is to make all subjects to have sequential labels for training for face identification
        mask: a mask to select a subset of samples for training
        
        """
        cframe=[]
        gf=[]
        cons_check_plist=[]
        cons_check_pfid=[]
        cons_check_final_pfid=[]
        upd_rate=[]
        final_updated_cons_dict={}
        final_updated_cons_dict_all=[]
        self.ufid=None
        self.cons_check_pfid=None
        self.upd_rate=None
        ratings=data_frame.iloc[:,3]       
        # rating_threshold=median(ratings) # median and mode is 3 for VM Subset, whereas mean is 3.2
        if C.filter_low_rating and not(order_rating):
            frame = data_frame.loc[lambda x: (x.iloc[:,3] == 0) | (x.iloc[:,3] > C.rating_threshold)] # Exclude sketches with ratings < rating_threshold
#            frame = data_frame.loc[lambda _df: (_df.iloc[:,3]==0) | (_df.iloc[:,3]>C.rating_threshold), :] # Exclude MIS sketches
        else:
            frame = data_frame
        if order_rating:  # For consistency check (of the model via human ratings)            
            frame_ = data_frame.loc[lambda x: (x.iloc[:,3] > 0)] # Take only the mismatched photos
            # frame_=frame_.iloc[:200] # Take only the first 100 rows from the meta data file # For quick test
            ufid=frame_.iloc[:,1].unique().tolist()
            gf=frame_.groupby(frame_.iloc[:,1])
            for i, jdx in enumerate(ufid):
                if (i==1):
                    print("Alert!")
                # if (jdx=='674558748_43') or (jdx=='am0151') or (jdx=='tsfwmale66-2neutral'):
                #     print("Alert!")
                if (jdx =='BM_PI_2_bmale0248f'):
                    print("Alert!")
                frame_jdx=gf.get_group(jdx)
                frame_jdx=frame_jdx.sort_values(by=3, ascending=False) # Sort 4th column
                cframe.append(frame_jdx)
                #********************************
                cons_check_plist.append(cframe[i][0].tolist()) # Retrieve the Photo file names
                dict_vals=cframe[i][3].tolist()
              
                pfname=cons_check_plist[i]   #Photo file names
                pfid=[]
                for idx in range(len(pfname)):
                    pfname_=pfname[idx].split(".",2)[0]
                    pfid_=pfname_.split("/",6)[-1]
                    pfid.append(pfid_)
            
                cons_check_pfid.append(pfid)
                # using naive method to convert two lists to dictionary
                cons_dict={}
                for key in cons_check_pfid[i]:
                    for value in dict_vals:
                        cons_dict[key]=value
                        dict_vals.remove(value)
                        break
                # To find the Normalized Discounted Cumulative Gain (NDCG) Score
                updated_cons_dict={}
                vals=[5,4,3,2,1]
                for key in cons_check_pfid[i]:
                    for value in vals :
                        updated_cons_dict[key]=value
                        vals.remove(value)
                        break
                final_updated_cons_dict[i]=updated_cons_dict
                # Do the below process to find the permutations based on the actual ratings (which contain duplicates)
                # Flipping dictionary key and values (due to duplicate values in the dict)
                flip_dict={}
                for key, val in cons_dict.items():
                    if val not in flip_dict:
                        flip_dict[val]=[key]
                    else:
                        flip_dict[val].append(key)
                fin_perlist=[]
                perlist=[]
                eqon=False
                gron=False
                # ccat=False
                for k,v in flip_dict.items():
                    plist=[]
                    plist1=[]
                    plist2=[]
                    if len(v) > 1:   # More than one negative photo got the same rating
                        gron=True
                        if (eqon and len(fin_perlist[0])<2):
                           dummy=fin_perlist
                           dummy = list(chain.from_iterable(dummy))  # Flattern 2D list
                           fin_perlist=[]
                           pval1=list(itertools.permutations(v))
                           for kk in pval1:
                               plist.append(list(kk))
                           for ndx in range(len(plist)):
                                   fin_perlist.append(dummy+plist[ndx])                         
                           # ccat=True
                        elif (eqon and len(fin_perlist[0])>2):
                           dummy=fin_perlist
                           fin_perlist=[]
                           pval1=list(itertools.permutations(v))
                           for kk in pval1:
                               plist.append(list(kk))
                           for d1 in range(len(dummy)):
                               for ndx in range(len(plist)):
                                   fin_perlist.append(dummy[d1]+plist[ndx])   
                        else:
                            pval=list(itertools.permutations(v))
                            for kdx in pval:
                                 plist1.append(list(kdx))
                            if not perlist:  # If List is empty
                                perlist=plist1
                                fin_perlist=perlist
                            else:
                                fin_perlist=[]
                                for mdx in range(len(perlist)):
                                  for ndx in range(len(plist1)):
                                      fin_perlist.append(perlist[mdx]+plist1[ndx])    
                            
                    if len(v) == 1:
                        eqon=True
                        if gron:
                            dummy=fin_perlist
                            fin_perlist=[]
                            plist2.append(v)
                            for mdx in range(len(dummy)):
                                for ndx in range(len(plist2)):
                                    fin_perlist.append(dummy[mdx]+plist2[ndx])    
                        else:
                            fin_perlist.append(v)
                # uniq_dummy_list=fin_perlist  # To retrieve the 2D list as it is (to compute NDCG)       
                #Final check
                for i in fin_perlist:
                    if len(i) == 1:
                        dummy_list=fin_perlist
                        fin_perlist=[]
                        dummy_list=list(chain.from_iterable(dummy_list))  # Flattern 2D list
                        fin_perlist.append(dummy_list)
                        break
                
                # To assign dummy ratings corresponding to the photoid to the permuted combinations
                updated_cons_dict_all={}
                upd_rate1=[]
                for row in fin_perlist:
                    upd_rate_=[]
                    for col in row:
                        if (col in updated_cons_dict.keys()):
                            upd_rate_.append(updated_cons_dict[col])
                    upd_rate1.append(upd_rate_)
                upd_rate.append(upd_rate1)
                
                final_updated_cons_dict_all_=[]
                for row in fin_perlist:
                    updated_cons_dict_all={}
                    vals1=[5,4,3,2,1]
                    for col in row:
                        for v1 in vals1:
                            updated_cons_dict_all[col]=v1 # To save all the permutations in a dictionary
                            vals1.remove(v1)
                            break
                    final_updated_cons_dict_all_.append(updated_cons_dict_all)
                final_updated_cons_dict_all.append(final_updated_cons_dict_all_)
                    
                print("\n\nUpdated dictionary for the composite id {} is {}:".format(jdx,updated_cons_dict))
                print("The permutations are:", fin_perlist)
                print("Updated dummy ratings for the permutations of this composite is:",upd_rate1)
                print("The length of the permutations for the composite id {0} is {1}".format(jdx, len(fin_perlist)))               
                cons_check_final_pfid.append(fin_perlist)  # Update the order of all possible  permutations for each identity
                
                frame=pd.concat(cframe)  # Sorted the ratings in descending order, if the fileids are same
            
            self.ufid=ufid  #1D list # Unique Composite ID
            # self.cons_check_pfid=cons_check_pfid
            self.cons_check_pfid=cons_check_final_pfid # 3D list # permutations of photo file ID for each identity
            self.final_updated_cons_dict=final_updated_cons_dict  # Assignment of dummy ratings to the initial order of photoids
            self.final_updated_cons_dict_all=final_updated_cons_dict_all # 2D list of dictionaries containing all possible permutations/identity
            self.upd_rate=upd_rate  # 3D list # Updated dummy ratings for the permuted combinations of all identities
            
        self.dir = frame.reset_index(drop=True)
#        self.dir = data_frame #pd.read_csv(C.meta_file, sep='@@@', header=None, engine='python')

        l1=[]
        f_dirpath=self.dir.iloc[:,0]    
        f_file_ids=self.dir.iloc[:,1]
        f_modality=self.dir.iloc[:,2]   # composite:0, photo:1
        f_ratings=self.dir.iloc[:,3]   # BM:5, MIS:3, PI: 4, GT:0
        
        idx1=0
        
        for idx1 in tqdm(range(len(f_dirpath))):
            
#            if (not C.load_setB) and (idx1 - 1) % 3 == 0: # skip set B composites
#                continue
#            modality = int(((idx1+1)%3 == 0)) # composite:0, photo:1
            f_dirpath1=f_dirpath[idx1]
            fileids1=f_file_ids[idx1]
            modality1=f_modality[idx1]
            ratings1=f_ratings[idx1]
#            #if C.bm_pi is True, uncomment the following
#            if "_MIS_" in f_dirpath1:
#                continue
#            #if C.bm_alone is True, uncomment the following
#            if "_MIS_" in f_dirpath1 or "_PI_" in f_dirpath1:
#                continue
            #if bm_pi_mis is True, don't do anything
            l1.append([f_dirpath1, fileids1, modality1,ratings1])
#            files1= [f1 for f1 in listdir(f_dirpath1) if isfile(join(f_dirpath1,f1))]
#            if len(files1) > 1:
#                # Sort files1 according to the number in the end of the file name
#                morph_id = [int(os.path.splitext(fn)[0].split('_')[-1]) for fn in files1]
#                sorted_ind = np.argsort(morph_id)
#                for i in sorted_ind:
#                    if C.sample_mask is None or C.sample_mask is not None and morph_id[i] in C.sample_mask:
#                        l1.append([f_dirpath1+files1[i], fileids1, modality])
#            else: # The subfolder only contains one example
#                l1.append([f_dirpath1+files1[0], fileids1, modality])
         
        data=l1  # 2444*6 = 14664 rows
        # Create the pandas DataFrame 
        self.frame = pd.DataFrame(data, columns = ['File_Name', 'File_Id', 'Modality', 'Ratings']) 
        
        self.label_map = None
        self.label_imap = None
        if C.reset_label: # Replace original labels (if they are string) with sequential integers
            labelencoder = LabelEncoder()
            new_labels = labelencoder.fit_transform(self.frame['File_Id'].values)
            self.frame['File_Id'] = new_labels
            self.label_map = {i:labelencoder.classes_[i] for i in range(len(labelencoder.classes_))} # Dictionary comprehension # Actual labels to integers
            self.label_imap = {labelencoder.classes_[i]:i for i in range(len(labelencoder.classes_))} # Dictionary comprehension # Integers to actual labels
        self.transform = self._get_data_transforms()    
        self.frame.sort_values(by=['File_Id', 'Modality'],  inplace=True)  # It's crucial to sort the data frame based on fileid and modality  
        self.filenames = self.frame.iloc[:,0].tolist()
        self.labels = None # numpy array
        self.label_set = None # list
        self.label_to_indices = None # dict where values are numpy arrays

        # The following initialization is for BalancedBatchSampler or its kind
        if self.frame.shape[1] > 1:   # if more than one row            
            self.labels = self.frame.iloc[:,1:4].to_numpy()  # Labels now include id, modality and ratings
            self.label_set = np.sort(np.unique(self.labels[:,0])).tolist()# It's crucial to sort the label_set
            self.label_to_indices = {}                    
            k = 0
            new_label_set=[]
            for label in self.label_set:
                indexes = []
                while k < len(self.labels) and self.labels[k,0] == label:
                    indexes.append(k)
                    k += 1
                if (len(indexes) > 1):
                   new_label_set.append(label)
                   self.label_to_indices[label] = np.array(indexes) # Dictionary comprehension
            self.label_set=new_label_set
            no_of_samples=sum([len(self.label_to_indices[i]) for i in self.label_set])
            print("Qualified no of identities are {0}".format(len(self.label_set)))
            print("Qualified no of samples is {0}".format(no_of_samples))
    def __getitem__(self, index):
        timg = []
        
        if type(index) in (tuple, list):
            if C.data_mix and C.data_mix_type == 'warp':
                pts = [np.loadtxt(os.path.splitext(self.filenames[i])[0]+'.pts') for i in index]
                pts = np.concatenate([np.array(x, dtype=np.float32)[np.newaxis,...] for x in pts])            
            if self.transform is not None:   
                timg = [self.transform(Image.open(self.filenames[i])) for i in index]
                if type(timg[0]) is torch.Tensor:
                    timg = torch.cat([x.unsqueeze(0) for x in timg])
                else:
                    timg = np.concatenate([np.array(x, dtype=np.float32)[np.newaxis,...] for x in timg])
                    timg = np.transpose(timg, axes=[0, 3, 1, 2])
            else:
                timg = [np.array(Image.open(self.filenames[i]), dtype=np.float32) for i in index]
                timg = np.concatenate([x[np.newaxis,...] for x in timg])
                timg = np.transpose(timg, axes=[0, 3, 1, 2])
                
        else:  # type(index) is integer
            if C.data_mix and C.data_mix_type == 'warp':
                pts = np.loadtxt(os.path.splitext(self.filenames[index])[0]+'.pts')
            if self.transform is not None:   
                 timg = self.transform(Image.open(self.filenames[index]))
#                if (C.predict_automatic_rating == False or C. existing_model == True):
#                    timg = self.transform(Image.open(self.filenames[index]))
#                else:
#                    #********************************************************************************************************************** 
#                    # For deploying the automatic rating model
#                    timg = self.transform(Image.open(self.filenames[index]))
##                    timg=np.array(timg)  # Convert image tensor into numpy array, in order to resize and convert into grayscale 
#                    timg = np.array(timg, dtype=np.float32)  # Convert image tensor into numpy array, in order to resize and convert into grayscale 
#                    timg = np.transpose(timg, axes=[1, 2, 0])
#                    timg=cv2.cvtColor(timg, cv2.COLOR_BGR2GRAY) # Convert into grayscale
#                    timg=cv2.resize(timg, (128,128))  # Resize the image
#                    timg=np.expand_dims(timg, axis=-1)
#                    
#                    #**********************************************************************************************************************
                 if type(timg) is not torch.Tensor:
                    timg = np.array(timg, dtype=np.float32)
                    timg = np.transpose(timg, axes=[2, 0, 1])
            else:
                timg = np.array(Image.open(self.filenames[index]), dtype=np.float32)
                timg = np.transpose(timg, axes=[2, 0, 1])
        label = None
        if self.labels is not None:
            label = self.labels[index]
            if C.data_mix and C.data_mix_type == 'warp':
                return (timg, pts, label), label
            else:
                return (timg, label), label
        else:
            if C.data_mix and C.data_mix_type == 'warp':
                return timg, pts
            else:
                return timg

    def __len__(self):
        return len(self.labels)
    
    def n_class(self):
        if self.label_set is not None:
            return len(self.label_set)
        else:
            return len(self.filenames)
    
    def get_labels(self):
        return self.labels
    def get_label_set(self):
        return self.label_set
    def get_label_to_indices(self):
        return self.label_to_indices
  
    def _get_data_transforms(self):
        return compose_transforms(C.meta, resize=C.data_transform['img_resize'],  
                                  to_grayscale=C.data_transform['to_grayscale'],
                                  crop_type=C.data_transform['crop_type'],
                                  override_meta_imsize=C.data_transform['override_meta_imsize'],
                                  random_flip=C.data_transform['random_flip']) 

class MUoMSGFSDataset(Dataset):
    """
    Morphed UoMSGFS Dataset with two labels: subject IDs and modality
    """
    def __init__(self, data_frame):
        """
        reset_label: the purpose is to make all subjects to have sequential labels for training for face identification
        mask: a mask to select a subset of samples for training
        """
        self.dir = data_frame #pd.read_csv(C.meta_file, sep='@@@', header=None, engine='python')

        l1=[]
        f_dirpath=self.dir.iloc[:,0]    # len(f_dirpath=1387)
        f_file_ids=self.dir.iloc[:,1]
        idx1=0
        
        for idx1 in tqdm(range(len(f_dirpath))):
            # if (not C.load_setA) and ((idx1+1) - 1) % 3 == 0: # skip set A composites
            #     continue
            if (not C.load_setB) and (idx1 - 1) % 3 == 0: # skip set B composites
                continue
            modality = int(((idx1+1)%3 == 0)) # composite:0, photo:1
            f_dirpath1=f_dirpath[idx1]
            fileids1=f_file_ids[idx1]
            # For morphed UoM-SGFS dataset
            files1= [f1 for f1 in listdir(f_dirpath1) if isfile(join(f_dirpath1,f1))]
            if len(files1) > 1:
                # Sort files1 according to the number in the end of the file name
                morph_id = [int(os.path.splitext(fn)[0].split('_')[-1]) for fn in files1]
                sorted_ind = np.argsort(morph_id)
                for i in sorted_ind:
                    if C.sample_mask is None or C.sample_mask is not None and morph_id[i] in C.sample_mask:
                        l1.append([f_dirpath1+files1[i], fileids1, modality])
            else: # The subfolder only contains one example
                l1.append([f_dirpath1+files1[0], fileids1, modality])
            # l1.append([f_dirpath1, fileids1, modality])  # For one photo per identity
         
        data=l1
        # Create the pandas DataFrame 
        self.frame = pd.DataFrame(data, columns = ['File_Name', 'File_Id', 'Modality']) 
        self.label_map = None
        self.label_imap=None
        if C.reset_label:
            labelencoder = LabelEncoder()
            new_labels = labelencoder.fit_transform(self.frame['File_Id'].values)
            self.frame['File_Id'] = new_labels
            self.label_map = {i:labelencoder.classes_[i] for i in range(len(labelencoder.classes_))}
            self.label_imap ={labelencoder.classes_[i]: i for i in range(len(labelencoder.classes_))}
        self.transform = self._get_data_transforms()               
        self.filenames = self.frame.iloc[:,0]
        self.labels = None # numpy array
        self.label_set = None # list
        self.label_to_indices = None # dict where values are numpy arrays

        # The following initialization is for BalancedBatchSampler or its kind
        if self.frame.shape[1] > 1:   # if more than one row            
            self.labels = self.frame.iloc[:,1:].to_numpy()
            self.label_set = np.sort(np.unique(self.labels[:,0])).tolist()         # It's crucial to sort the label_set
            self.label_to_indices = {}                    
            k = 0
            new_label_set=[]
            for label in self.label_set:
                indexes = []
                while k < len(self.labels) and self.labels[k,0] == label:
                    indexes.append(k)
                    k += 1
                if (len(indexes) > 1):
                    new_label_set.append(label)
                    self.label_to_indices[label] = np.array(indexes) # Dictionary Comprehension
            self.label_set=new_label_set
            no_of_samples=sum([len(self.label_to_indices[i]) for i in self.label_set])
            print("Qualified no of identities are {0}".format(len(self.label_set)))
            print("Qualified no of samples is {0}".format(no_of_samples))
    def __getitem__(self, index):
        timg = []
        
        if type(index) in (tuple, list):
            if C.data_mix and C.data_mix_type == 'warp':
                pts = [np.loadtxt(os.path.splitext(self.filenames[i])[0]+'.pts') for i in index]
                pts = np.concatenate([np.array(x, dtype=np.float32)[np.newaxis,...] for x in pts])            
            if self.transform is not None:   
                timg = [self.transform(Image.open(self.filenames[i])) for i in index]
                if type(timg[0]) is torch.Tensor:
                    timg = torch.cat([x.unsqueeze(0) for x in timg])
                else:
                    timg = np.concatenate([np.array(x, dtype=np.float32)[np.newaxis,...] for x in timg])
                    timg = np.transpose(timg, axes=[0, 3, 1, 2])
            else:
                timg = [np.array(Image.open(self.filenames[i]), dtype=np.float32) for i in index]
                timg = np.concatenate([x[np.newaxis,...] for x in timg])
                timg = np.transpose(timg, axes=[0, 3, 1, 2])
                
        else:
            if C.data_mix and C.data_mix_type == 'warp':
                pts = np.loadtxt(os.path.splitext(self.filenames[index])[0]+'.pts')
            if self.transform is not None:           
                timg = self.transform(Image.open(self.filenames[index]))
                if type(timg) is not torch.Tensor:
                    timg = np.array(timg, dtype=np.float32)
                    timg = np.transpose(timg, axes=[2, 0, 1])
            else:
                timg = np.array(Image.open(self.filenames[index]), dtype=np.float32)
                timg = np.transpose(timg, axes=[2, 0, 1])
           
        label = None
        if self.labels is not None:
            label = self.labels[index]
            if C.data_mix and C.data_mix_type == 'warp':
                return (timg, pts, label), label
            else:
                return (timg, label), label
        else:
            if C.data_mix and C.data_mix_type == 'warp':
                return timg, pts
            else:
                return timg

    def __len__(self):
        return len(self.labels)
    
    def n_class(self):
        if self.label_set is not None:
            return len(self.label_set)
        else:
            return len(self.filenames)
    
    def get_labels(self):
        return self.labels
    
    def get_label_to_indices(self):
        return self.label_to_indices
  
    def _get_data_transforms(self):
        return compose_transforms(C.meta, resize=C.data_transform['img_resize'], 
                                  to_grayscale=C.data_transform['to_grayscale'],
                                  crop_type=C.data_transform['crop_type'],
                                  override_meta_imsize=C.data_transform['override_meta_imsize'],
                                  random_flip=C.data_transform['random_flip']) 

class MUoMSGFSADataset(Dataset):
    """
    Morphed UoMSGFS Dataset with attributes: gender and race
    """
    def __init__(self, data_frame, attr_frame):
        """
        reset_label: the purpose is to make all subjects to have sequential labels for training for face identification
        mask: a mask to select a subset of samples for training
        """
        self.dir = data_frame #pd.read_csv(C.meta_file, sep='@@@', header=None, engine='python')

        l1=[]
        f_dirpath=self.dir.iloc[:,0]    # len(f_dirpath=1387)
        f_file_ids=self.dir.iloc[:,1]
        one_hot_encoder = OneHotEncoder()
        gender = one_hot_encoder.fit_transform(attr_frame.iloc[:,2].values.reshape(-1, 1)).toarray().astype(dtype=np.float32)
        race = one_hot_encoder.fit_transform(attr_frame.iloc[:,3].values.reshape(-1, 1)).toarray().astype(dtype=np.float32)
        sid_to_idx = {int(attr_frame.iloc[:, 0].values[i][1:-1]): i for i, sid in enumerate(attr_frame.iloc[:, 0].values)}
        idx1=0
        
        for idx1 in tqdm(range(len(f_dirpath))):
            if (not C.load_setB) and (idx1 - 1) % 3 == 0: # skip set B composites
                continue
            modality = int(((idx1+1)%3 == 0)) # composite:0, photo:1
            f_dirpath1=f_dirpath[idx1]
            fileids1=f_file_ids[idx1]
            _gender = gender[sid_to_idx[fileids1]]
            _race = race[sid_to_idx[fileids1]]          
            files1= [f1 for f1 in listdir(f_dirpath1) if isfile(join(f_dirpath1,f1))]
            if len(files1) > 1:
                # Sort files1 according to the number in the end of the file name
                morph_id = [int(os.path.splitext(fn)[0].split('_')[-1]) for fn in files1]
                sorted_ind = np.argsort(morph_id)
                for i in sorted_ind:
                    if C.sample_mask is None or C.sample_mask is not None and morph_id[i] in C.sample_mask:
                        l1.append([f_dirpath1+files1[i], fileids1, modality, _gender, _race])
            else: # The subfolder only contains one example
                l1.append([f_dirpath1+files1[0], fileids1, modality, _gender, _race])
         
        data=l1
        # Create the pandas DataFrame 
        self.frame = pd.DataFrame(data, columns = ['File_Name', 'File_Id', 'Modality', 'Gender', 'Race']) 
        self.label_map = None
        if C.reset_label:
            labelencoder = LabelEncoder()
            new_labels = labelencoder.fit_transform(self.frame['File_Id'].values)
            self.frame['File_Id'] = new_labels
            self.label_map = {i:labelencoder.classes_[i] for i in range(len(labelencoder.classes_))}
            
        self.transform = self._get_data_transforms()               
        self.filenames = self.frame.iloc[:,0]
        self.labels = None # numpy array
        self.label_set = None # list
        self.label_to_indices = None # dict where values are numpy arrays

        # The following initialization is for BalancedBatchSampler or its kind
        if self.frame.shape[1] > 1:   # if more than one row            
            self.labels = self.frame.iloc[:, 1:3].to_numpy()
            self.gender = np.vstack(self.frame.iloc[:, 3].to_numpy())
            self.race = np.vstack(self.frame.iloc[:, 4].to_numpy())
            self.label_set = np.sort(np.unique(self.labels[:,0])).tolist()         # It's crucial to sort the label_set
            self.label_to_indices = {}                    
            k = 0
            
            for label in self.label_set:
                indexes = []
                while k < len(self.labels) and self.labels[k,0] == label:
                    indexes.append(k)
                    k += 1
                self.label_to_indices[label] = np.array(indexes)
       
    def __getitem__(self, index):
        timg = []
        
        if type(index) in (tuple, list):
            if self.transform is not None:   
                timg = [self.transform(Image.open(self.filenames[i])) for i in index]
                if type(timg[0]) is torch.Tensor:
                    timg = torch.cat([x.unsqueeze(0) for x in timg])
                else:
                    timg = np.concatenate([np.array(x, dtype=np.float32)[np.newaxis,...] for x in timg])
                    timg = np.transpose(timg, axes=[0, 3, 1, 2])
            else:
                timg = [np.array(Image.open(self.filenames[i]), dtype=np.float32) for i in index]
                timg = np.concatenate([x[np.newaxis,...] for x in timg])
                timg = np.transpose(timg, axes=[0, 3, 1, 2])
        else:
            if self.transform is not None:           
                timg = self.transform(Image.open(self.filenames[index]))
                if type(timg) is not torch.Tensor:
                    timg = np.array(timg, dtype=np.float32)
                    timg = np.transpose(timg, axes=[2, 0, 1])
            else:
                timg = np.array(Image.open(self.filenames[index]), dtype=np.float32)
                timg = np.transpose(timg, axes=[2, 0, 1])
           
        label = None
        if self.labels is not None:
            label = self.labels[index]
            gender = self.gender[index]
            race = self.race[index]
            return timg, (label, gender, race)
        else:
            return timg

    def __len__(self):
        return len(self.labels)
    
    def n_class(self):
        if self.label_set is not None:
            return len(self.label_set)
        else:
            return len(self.filenames)
    
    def get_labels(self):
        return self.labels
    
    def get_label_to_indices(self):
        return self.label_to_indices
  
    def _get_data_transforms(self):
        return compose_transforms(C.meta, resize=C.data_transform['img_resize'], 
                                  to_grayscale=C.data_transform['to_grayscale'],
                                  crop_type=C.data_transform['crop_type'],
                                  override_meta_imsize=C.data_transform['override_meta_imsize'],
                                  random_flip=C.data_transform['random_flip']) 

#class VMBatchSampler(BatchSampler):
#    """
#    Stratified sampling - from a MNIST-like dataset, samples n_classes and for each of these classes samples n_sample_per_class.
#    Returns batches of size n_classes * n_sample_per_class. For each class, half of the samples are setA composites
#    and half are photos.
#    """
#    def __init__(self, labels, label_to_indices, label_set, n_classes_per_batch, n_samples_per_class):
#        if type(labels) in (tuple, list):
#            labels = np.array(labels)
#        self.labels = labels # np.ndarray
#        if isinstance(label_set, np.ndarray):
#            label_set = label_set.tolist()
#        self.label_set = label_set # list
#        self.label_to_indices = label_to_indices # dict of values of type np.ndarray
#        self.used_lbl_idx_cnt = {label: [0,0] for label in self.label_set}
#        self.count = 0
#        self.n_classes = n_classes_per_batch    #32
#        self.n_samples = n_samples_per_class//2 * 2 # make sure it is an even number
#        
#        self.n_class_batch = len(label_set) // self.n_classes # Actual
#        
#        if len(label_set) % self.n_classes > 0:  
#            self.n_class_batch += 1
#        #self.n_triplet = len(label_set) * (len(label_set) - 1) * 100 # number of triplets  (train-(432*431=186,192 triplets) and val -2256 triplets)                  
#        self.batch_size = self.n_classes * self.n_samples # 192 for 1-BM and 5-PIs
#        self.sample_count = sum([len(label_to_indices[x]) for x in label_set])  
#        
#    def __iter__(self):
#        self.count = 0
#        i = 0
#        while self.count + self.batch_size <= self.sample_count:  # Actual
#            if len(self.label_set) > self.n_classes: 
#                #classes = np.random.choice(self.label_set, self.n_classes, replace=False)  # Generates n_classes(32)  of random sample from the label_set (432 or 48)
#                i = i % self.n_class_batch # Actual
#                if i == 0:
#                    np.random.shuffle(self.label_set)
#                if i == self.n_class_batch - 1:  # i==61
#                    classes = self.label_set[i*self.n_classes:]
##                    classes += self.label_set[:self.n_classes - len(classes)]  # For a complete batch of 32  # Commented by Siva
#                    # If the number of classes in the last batch is less than n_classes, add the classes from the first batch
#                else:
#                    classes = self.label_set[i*self.n_classes:(i+1)*self.n_classes]
#                i += 1
#            else:
#                classes = list(self.label_set)  
#            indices = []
#            
#            for class_ in classes:
#                ltoi  = self.label_to_indices[class_] # images with same file_id
#                if not isinstance(ltoi, np.ndarray):
#                    ltoi  = np.array(ltoi)
#                compo_indices = ltoi[self.labels[ltoi,1]==0].tolist() # 0 - Composite
#                photo_indices = ltoi[self.labels[ltoi,1]==1].tolist() # 1 - Photo
#                
#                
#                headpos = self.used_lbl_idx_cnt[class_][0]  # 0
#                if (C.bm_alone):
#                    n_bcompo = self.n_samples//2   # 1  # For equal number of composites and photo
#                else:
#                    n_bcompo=self.n_samples-1   # For multiple composites and one photo
#                tailpos = headpos + n_bcompo   # 1
#                if len(compo_indices) > 0:
#                    if tailpos > len(compo_indices):
#                        n_bcompo = min(n_bcompo, len(compo_indices))
#                        indices.extend(np.random.choice(compo_indices, n_bcompo, replace=False))
#                    else:
#                        indices.extend(compo_indices[headpos:tailpos])
#                headpos = self.used_lbl_idx_cnt[class_][1]
#                if (C.bm_alone):
#                    n_bphoto = self.n_samples//2   # For equal number of composites and photo
#                else:
#                    n_bphoto = self.n_samples-1  # For multiple composites and one photo
#                tailpos = headpos + n_bphoto
#                if len(photo_indices) > 0:
#                    if tailpos > len(photo_indices):
#                        n_bphoto = min(n_bphoto, len(photo_indices))
#                        indices.extend(np.random.choice(photo_indices, n_bphoto, replace=False))
#                    else:
#                        indices.extend(photo_indices[headpos:tailpos])
#                if self.used_lbl_idx_cnt[class_][0] < len(compo_indices):
#                    self.used_lbl_idx_cnt[class_][0] += n_bcompo   #increment the composite pointer
#                if self.used_lbl_idx_cnt[class_][1] < len(photo_indices):
#                    self.used_lbl_idx_cnt[class_][1] += n_bphoto  #increment the photo pointer
#                if sum(self.used_lbl_idx_cnt[class_]) >= len(self.label_to_indices[class_]):
#                    np.random.shuffle(self.label_to_indices[class_])
#                    self.used_lbl_idx_cnt[class_] = [0, 0]  #reset the pointers back to 0
#                    
#            yield indices      # Used to iterate over a sequence, but don't want to store the entire sequence in memory
#                          
##            self.count += self.batch_size  # Next execution resumes from this point  # Actual
#            
#             # Added by Siva
#            self.count += self.batch_size  # Next execution resumes from this point 
#            if (self.sample_count == self.count):
#                break
#            elif (self.sample_count - self.count < self.batch_size):
#                self.batch_size=self.sample_count - self.count 
#           
#
#    def __len__(self):
##        return self.n_triplet // self.batch_size    # No of batches to sample all the triplets (187 - to train and 3 - to val)
#        if ((self.sample_count // self.batch_size) % 2 == 0): # Check for non-zero remainder, if not add 1 to n_batch
#            n_batch = self.sample_count // self.batch_size
#        else:
#            n_batch = self.sample_count // self.batch_size + 1 
#            
##        n_batch = self.sample_count // self.batch_size  # Actual
#        return n_batch


class MUoMSGFSBatchSampler(BatchSampler):
    """
    Stratified sampling - from a MNIST-like dataset, samples n_classes and for each of these classes samples n_sample_per_class.
    Returns batches of size n_classes * n_sample_per_class. For each class, half of the samples are setA composites
    and half are photos.
    """
    def __init__(self, labels, label_to_indices, label_set, n_classes_per_batch, n_samples_per_class):
        if type(labels) in (tuple, list):
            labels = np.array(labels)
        self.labels = labels # np.ndarray
        if isinstance(label_set, np.ndarray):
            label_set = label_set.tolist()
        self.label_set = label_set # list
        
        self.label_to_indices = label_to_indices # dict of values of type np.ndarray
        self.used_lbl_idx_cnt = {label: [0,0] for label in self.label_set}
        self.count = 0
        self.n_classes = n_classes_per_batch    #32
        self.n_samples = n_samples_per_class//2 * 2 # make sure it is an even number
        
        self.n_class_batch = len(label_set) // self.n_classes # Actual
        
        if len(label_set) % self.n_classes > 0:  
            self.n_class_batch += 1
        #self.n_triplet = len(label_set) * (len(label_set) - 1) * 100 # number of triplets  (train-(432*431=186,192 triplets) and val -2256 triplets)                  
        self.batch_size = self.n_classes * self.n_samples # 192 for 1-BM and 5-PIs
        # self.sample_count=0
        # for x in self.label_set:
        #       self.sample_count+=len(self.label_to_indices[x])
        self.sample_count = sum([len(self.label_to_indices[x]) for x in self.label_set])  
        
    def __iter__(self):
        self.count = 0
        i = 0
        while self.count + self.batch_size <= self.sample_count:  # Actual
            if len(self.label_set) > self.n_classes: 
                #classes = np.random.choice(self.label_set, self.n_classes, replace=False)  # Generates n_classes(32)  of random sample from the label_set (432 or 48)
                i = i % self.n_class_batch # Actual
                if i == 0:
                    np.random.shuffle(self.label_set)
                if i == self.n_class_batch - 1:  # i==61
                    classes = self.label_set[i*self.n_classes:]
                    classes += self.label_set[:self.n_classes - len(classes)]  # For a complete batch of 32        # Commented by Siva
                    # If the number of classes in the last batch is less than n_classes, add the classes from the first batch
                else:
                    classes = self.label_set[i*self.n_classes:(i+1)*self.n_classes]
                i += 1
            else:
                classes = list(self.label_set)  
            indices = []
            
            for class_ in classes:
                ltoi  = self.label_to_indices[class_] # images with same file_id
                if not isinstance(ltoi, np.ndarray):
                    ltoi  = np.array(ltoi)
                compo_indices = ltoi[self.labels[ltoi,1]==0].tolist() # 0 - Composite
                photo_indices = ltoi[self.labels[ltoi,1]==1].tolist() # 1 - Photo
                
                
                headpos = self.used_lbl_idx_cnt[class_][0]  # 0
                if (C.bm_alone):
                    n_bcompo = self.n_samples//2   # 1  # For BM alone
                else:
                    n_bcompo = self.n_samples-1  # For BM or PI or MIS or all
                tailpos = headpos + n_bcompo   # 1
                if len(compo_indices) > 0:
                    if tailpos > len(compo_indices):
                        n_bcompo = min(n_bcompo, len(compo_indices))
                        indices.extend(np.random.choice(compo_indices, n_bcompo, replace=False))
                    else:
                        indices.extend(compo_indices[headpos:tailpos])
                headpos = self.used_lbl_idx_cnt[class_][1]
                if (C.bm_alone):
                    n_bphoto = self.n_samples//2    # For BM alone
                else:
                    n_bphoto = self.n_samples-1  # For BM or PI or MIS or all
                    
                tailpos = headpos + n_bphoto
                if len(photo_indices) > 0:
                    if tailpos > len(photo_indices):
                        n_bphoto = min(n_bphoto, len(photo_indices))
                        indices.extend(np.random.choice(photo_indices, n_bphoto, replace=False))
                    else:
                        indices.extend(photo_indices[headpos:tailpos])
                if self.used_lbl_idx_cnt[class_][0] < len(compo_indices):
                    self.used_lbl_idx_cnt[class_][0] += n_bcompo   #increment the composite pointer
                if self.used_lbl_idx_cnt[class_][1] < len(photo_indices):
                    self.used_lbl_idx_cnt[class_][1] += n_bphoto  #increment the photo pointer
                if sum(self.used_lbl_idx_cnt[class_]) >= len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_lbl_idx_cnt[class_] = [0, 0]  #reset the pointers back to 0
                    
            yield indices      # Used to iterate over a sequence, but don't want to store the entire sequence in memory
                          
            self.count += self.batch_size  # Next execution resumes from this point  # Actual
            
#             # Added by Siva
#            self.count += self.batch_size  # Next execution resumes from this point 
#            if (self.sample_count == self.count):
#                break
#            elif (self.sample_count - self.count < self.batch_size):
#                self.batch_size=self.sample_count - self.count 
           

    def __len__(self):
#        return self.n_triplet // self.batch_size    # No of batches to sample all the triplets (187 - to train and 3 - to val)
#        if ((self.sample_count // self.batch_size) % 2 == 0): # Check for non-zero remainder, if not add 1 to n_batch
#            n_batch = self.sample_count // self.batch_size
#        else:
#            n_batch = self.sample_count // self.batch_size + 1 
            
        n_batch = self.sample_count // self.batch_size  # Actual
        return n_batch



#class Prev_MUoMSGFSBatchSampler(BatchSampler):
#    """
#    Stratified sampling - from a MNIST-like dataset, samples n_classes and for each of these classes samples n_sample_per_class.
#    Returns batches of size n_classes * n_sample_per_class. For each class, half of the samples are setA composites
#    and half are photos.
#    """
#    def __init__(self, labels, label_to_indices, label_set, n_classes_per_batch, n_samples_per_class):
#        if type(labels) in (tuple, list):
#            labels = np.array(labels)
#        self.labels = labels # np.ndarray
#        if isinstance(label_set, np.ndarray):
#            label_set = label_set.tolist()
#        self.label_set = label_set # list
#        self.label_to_indices = label_to_indices # dict of values of type np.ndarray
#        self.used_lbl_idx_cnt = {label: [0,0] for label in self.label_set}
#        self.count = 0
#        self.n_classes = n_classes_per_batch    #32
#        self.n_samples = n_samples_per_class//2 * 2 # make sure it is an even number
#        
##        self.sample_count = sum([len(label_to_indices[x]) for x in label_set])
##        self.batch_size = self.n_classes * self.n_samples
#        self.n_class_batch = len(label_set) // self.n_classes # Actual
##        self.n_class_batch = self.sample_count //  self.batch_size  # Added by Siva
#        
#        if len(label_set) % self.n_classes > 0:  
#            self.n_class_batch += 1
#        #self.n_triplet = len(label_set) * (len(label_set) - 1) * 100 # number of triplets  (train-(432*431=186,192 triplets) and val -2256 triplets)                  
#        self.batch_size = self.n_classes * self.n_samples
#        self.sample_count = sum([len(label_to_indices[x]) for x in label_set])
#       
#    def __iter__(self):
#        self.count = 0
#        i = 0
#        while self.count + self.batch_size <= self.sample_count:  # Actual
##        while i < self.n_class_batch:
#            if len(self.label_set) > self.n_classes:
#                #classes = np.random.choice(self.label_set, self.n_classes, replace=False)  # Generates n_classes(32)  of random sample from the label_set (432 or 48)
#                i = i % self.n_class_batch
#                if i == 0:
#                    np.random.shuffle(self.label_set)
#                if i == self.n_class_batch - 1:  # i==61
#                    classes = self.label_set[i*self.n_classes:]
#                    classes += self.label_set[:self.n_classes - len(classes)]  # For a complete batch of 32        # Commented by Siva
#                    # If the number of classes in the last batch is less than n_classes, add the classes from the first batch
#                else:
#                    classes = self.label_set[i*self.n_classes:(i+1)*self.n_classes]
#                i += 1
#            else:
#                classes = list(self.label_set)  
#            indices = []
#            
#            for class_ in classes:
#                ltoi  = self.label_to_indices[class_] # images with same file_id
#                if not isinstance(ltoi, np.ndarray):
#                    ltoi  = np.array(ltoi)
#                compo_indices = ltoi[self.labels[ltoi,1]==0].tolist() # 0 - Composite
#                photo_indices = ltoi[self.labels[ltoi,1]==1].tolist() # 1 - Photo
#                
#                
#                headpos = self.used_lbl_idx_cnt[class_][0]  # 0
#                n_bcompo = self.n_samples//2   # 1
#                tailpos = headpos + n_bcompo   # 1
#                if len(compo_indices) > 0:
#                    if tailpos > len(compo_indices):
#                        n_bcompo = min(n_bcompo, len(compo_indices))
#                        indices.extend(np.random.choice(compo_indices, n_bcompo, replace=False))
#                    else:
#                        indices.extend(compo_indices[headpos:tailpos])
#                headpos = self.used_lbl_idx_cnt[class_][1]
#                n_bphoto = self.n_samples//2
#                tailpos = headpos + n_bphoto
#                if len(photo_indices) > 0:
#                    if tailpos > len(photo_indices):
#                        n_bphoto = min(n_bphoto, len(photo_indices))
#                        indices.extend(np.random.choice(photo_indices, n_bphoto, replace=False))
#                    else:
#                        indices.extend(photo_indices[headpos:tailpos])
#                if self.used_lbl_idx_cnt[class_][0] < len(compo_indices):
#                    self.used_lbl_idx_cnt[class_][0] += n_bcompo   #increment the composite pointer
#                if self.used_lbl_idx_cnt[class_][1] < len(photo_indices):
#                    self.used_lbl_idx_cnt[class_][1] += n_bphoto  #increment the photo pointer
#                if sum(self.used_lbl_idx_cnt[class_]) >= len(self.label_to_indices[class_]):
#                    np.random.shuffle(self.label_to_indices[class_])
#                    self.used_lbl_idx_cnt[class_] = [0, 0]  #reset the pointers back to 0
#                    
#            yield indices      # Used to iterate over a sequence, but don't want to store the entire sequence in memory
#                          
#            self.count += self.batch_size  # Next execution resumes from this point  # Actual
#            
##             # Added by Siva
##            self.count += self.batch_size  # Next execution resumes from this point 
##            if (self.sample_count == self.count):
##                break
##            elif (self.sample_count - self.count < self.batch_size):
##                self.batch_size=self.sample_count - self.count 
#           
#
#    def __len__(self):
##        return self.n_triplet // self.batch_size    # No of batches to sample all the triplets (187 - to train and 3 - to val)
##        if ((self.sample_count // self.batch_size) % 2 == 0): # Check for non-zero remainder, if not add 1 to n_batch
##            n_batch = self.sample_count // self.batch_size
##        else:
##            n_batch = self.sample_count // self.batch_size + 1 
#            
#        n_batch = self.sample_count // self.batch_size  # Actual
#        return n_batch
#        
class MUoMSGFSSequentialSampler(BatchSampler):
    """
    The test data are sampled from morphed dataset
    """

    def __init__(self, labels, label_to_indices, label_set, n_classes):
        self.labels = labels
        self.label_set = label_set 
        self.label_to_indices = label_to_indices
        self.used_label_indices_count = {label: 0 for label in self.label_set}
        self.count = 0
        self.n_classes = n_classes
        self.batch_size = self.n_classes * 3
        self.n_class_batch = len(label_set) // self.n_classes
        if len(label_set) % self.n_classes > 0:
            self.n_class_batch += 1
                
    def __iter__(self):
        self.count = 0
        setA_list = list(range(645))
        setB_list = list(range(645, 2*645))
        photo_list = list(range(2*645, 3*645))
        
        for i in range(self.n_class_batch):
            end_idx = (i+1)*self.n_classes
            if end_idx > len(self.label_set):
                end_idx = len(self.label_set)
            classes = self.label_set[i*self.n_classes:end_idx]
            indices = []
            
            for class_ in classes:
                indices.append(self.label_to_indices[class_][setA_list[313]])
                indices.append(self.label_to_indices[class_][setB_list[313]])
                indices.append(self.label_to_indices[class_][photo_list[313]])
                    
            yield indices


    def __len__(self):
        return self.n_class_batch    # No of batches to sample all the triplets (187 - to train and 3 - to val)

    
class SequentialSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(self.data_source)

    def __len__(self):
        return len(self.data_source)
    
class SMVMLoader(object):
    """
    Stratified Morphed UoMSGFS data loader firstly splits all samples into train&valididation 
    subset and test subset so that there is no overlapping identity
    among the two subsets. It then splits train&validation subset into train and
    validation subsets in such a way that both subsets contain the same identities
    but different samples from each identity as a result of stratification upon
    identities. The loader supports training with cross entropy losses where face
    identification is required.
    """
    def __init__(self):
        self.dataset, self.test_dataset, self.dataset_eg = self._get_dataset()
        num_class= self.dataset.n_class()    # 2444
        classes = self.dataset.label_set
        np.random.shuffle(classes)
        train_valid_percent = 1.0
        if not C.fixed_testset:
            train_valid_percent = 1.0 - C.test_set_size
        split = int(np.floor(train_valid_percent * num_class))
        train_valid_cls, test_cls = classes[:split], classes[split:]
        
        # Split the sample list of every subject to train and valid parts. The composites and photos
        # must be split separately.
        labels = self.dataset.get_labels()
        ltoi = self.dataset.get_label_to_indices()
        ltoi = {cl:[ltoi[cl][labels[ltoi[cl],1]==0], ltoi[cl][labels[ltoi[cl],1]==1]] for cl in train_valid_cls}
        for cl in train_valid_cls:
            np.random.shuffle(ltoi[cl][0])
            np.random.shuffle(ltoi[cl][1])
        train_ltoi = {}
        valid_ltoi = {}   
        
        for cl in train_valid_cls:
            train_size_1 = math.floor(len(ltoi[cl][0]) * (1 - C.valid_set_size))
            if train_size_1 == 0:
                train_size_1 = np.random.randint(0, 1)
            train_size_2 = math.ceil(len(ltoi[cl][1]) * (1 - C.valid_set_size))  # Changed here
            if train_size_2 == 0:
                train_size_2 = np.random.randint(0, 1)
            train_ltoi[cl] = [ltoi[cl][0][:train_size_1], ltoi[cl][1][:train_size_2]]
            valid_ltoi[cl] = [ltoi[cl][0][train_size_1:], ltoi[cl][1][:train_size_2]]  # Changed here
            
        train_ltoi = {cl:np.concatenate(train_ltoi[cl]) for cl in train_valid_cls}
        valid_ltoi = {cl:np.concatenate(valid_ltoi[cl]) for cl in train_valid_cls}
        
        train_sampler = MUoMSGFSBatchSampler(labels, 
                                            train_ltoi,
                                            train_valid_cls, C.n_bclass, C.n_sample_per_cls)
        valid_sampler = MUoMSGFSBatchSampler(labels,
                                            valid_ltoi,
                                            train_valid_cls, C.n_bclass, C.n_sample_per_cls)
        # Dataloader for training dataset
        self.train_loader = torch.utils.data.DataLoader(
            self.dataset, batch_sampler=train_sampler, shuffle=False,
            num_workers=C.n_worker, pin_memory=C.pin_memory
        )
        self.valid_loader = torch.utils.data.DataLoader(
            self.dataset, batch_sampler=valid_sampler,
            num_workers=C.n_worker, pin_memory=C.pin_memory
        )
        # Dataloader for test dataset
        classes_query = set(self.test_dataset.label_set)
        if test_cls:
            old_test_cls = test_cls            
            if C.reset_label:
                old_test_cls = [self.dataset.label_map[c] for c in test_cls]
            classes_query = list(classes_query.intersection(set(old_test_cls)))

        class_to_indices = self.test_dataset.get_label_to_indices()
        test_indices = [np.array(class_to_indices[clas]) for clas in classes_query]
        test_indices = np.concatenate(test_indices).tolist()        
        base_sampler = SequentialSampler(test_indices)
        test_sampler = BatchSampler(base_sampler, batch_size=32, drop_last=False)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, shuffle=False, batch_sampler=test_sampler,
            num_workers=C.n_worker, pin_memory=C.pin_memory,
        )
        # Dataloader for extended gallery
        eg_indices = list(range(len(self.dataset_eg)))   
        base_sampler = SequentialSampler(eg_indices)
        test_sampler = BatchSampler(base_sampler, batch_size=32, drop_last=False)
        self.eg_loader = torch.utils.data.DataLoader(
            self.dataset_eg, shuffle=False, batch_sampler=test_sampler,
            num_workers=C.n_worker, pin_memory=C.pin_memory,
        )
        

    def _get_dataset(self):
        train_df = pd.read_csv(C.train_meta_file, sep='@@@', header=None, engine='python')
        train_dataset = MVMDataset(train_df)
        test_df = pd.read_csv(C.test_meta_file, sep='@@@', header=None, engine='python')
        test_dataset = UoMSGFSDataset(test_df)
        eg_df = pd.read_csv(C.eg_meta_file, sep='@@@', header=None, engine='python')
        eg_dataset = UoMSGFSDataset(eg_df)
        
        return train_dataset, test_dataset, eg_dataset

    def _get_dataset_expt(self):
        raise NotImplementedError

    def _get_weighted_random_sampler(self, dataset, weight_column):
        """
        Args:
            dataset (torch.utils.data.Dataset):
            weight_column (str):

        Returns:

        """
        raise NotImplementedError
    
class SMUoMSGFSLoader(object):
    """
    Stratified Morphed UoMSGFS data loader firstly splits all samples into train&valididation 
    subset and test subset so that there is no overlapping identity
    among the two subsets. It then splits train&validation subset into train and
    validation subsets in such a way that BOTH SUBSETS CONTAIN THE SAME IDENTITIES
    but different samples from each identity as a result of stratification upon
    identities. The loader supports training with cross entropy losses where face
    identification is required.
    """
    def __init__(self):
        self.dataset, self.test_dataset, self.dataset_eg = self._get_dataset()
        num_class= self.dataset.n_class()    # 467
        classes = self.dataset.label_set
        np.random.shuffle(classes)
        train_valid_percent = 1.0
        if not C.fixed_testset:
            train_valid_percent = 1.0 - C.test_set_size
        split = int(np.floor(train_valid_percent * num_class))
        train_valid_cls, test_cls = classes[:split], classes[split:]
        
        # Split the sample list of very subject to train and valid parts. The composites and photos
        # must be split separately.
        labels = self.dataset.get_labels()
        ltoi = self.dataset.get_label_to_indices()
        ltoi = {cl:[ltoi[cl][labels[ltoi[cl],1]==0], ltoi[cl][labels[ltoi[cl],1]==1]] for cl in train_valid_cls}
        for cl in train_valid_cls:
            np.random.shuffle(ltoi[cl][0])
            np.random.shuffle(ltoi[cl][1])
        train_ltoi = {}
        valid_ltoi = {}   
        
        for cl in train_valid_cls:
            train_size_1 = math.floor(len(ltoi[cl][0]) * (1 - C.valid_set_size))
            if train_size_1 == 0:
                train_size_1 = np.random.randint(0, 1)
            train_size_2 = math.floor(len(ltoi[cl][1]) * (1 - C.valid_set_size))
            if train_size_2 == 0:
                train_size_2 = np.random.randint(0, 1)
            train_ltoi[cl] = [ltoi[cl][0][:train_size_1], ltoi[cl][1][:train_size_2]]
            valid_ltoi[cl] = [ltoi[cl][0][train_size_1:], ltoi[cl][1][train_size_2:]]
            
        train_ltoi = {cl:np.concatenate(train_ltoi[cl]) for cl in train_valid_cls}
        valid_ltoi = {cl:np.concatenate(valid_ltoi[cl]) for cl in train_valid_cls}
        
        train_sampler = MUoMSGFSBatchSampler(labels, 
                                            train_ltoi,
                                            train_valid_cls, C.n_bclass, C.n_sample_per_cls)
        valid_sampler = MUoMSGFSBatchSampler(labels,
                                            valid_ltoi,
                                            train_valid_cls, C.n_bclass, C.n_sample_per_cls)
        # Dataloader for training dataset
        self.train_loader = torch.utils.data.DataLoader(
            self.dataset, batch_sampler=train_sampler, shuffle=False,
            num_workers=C.n_worker, pin_memory=C.pin_memory
        )
        self.valid_loader = torch.utils.data.DataLoader(
            self.dataset, batch_sampler=valid_sampler,
            num_workers=C.n_worker, pin_memory=C.pin_memory
        )
        # Dataloader for test dataset
        classes_query = set(self.test_dataset.label_set)
        if test_cls:
            old_test_cls = test_cls            
            if C.reset_label:
                old_test_cls = [self.dataset.label_map[c] for c in test_cls]
            classes_query = list(classes_query.intersection(set(old_test_cls)))

        class_to_indices = self.test_dataset.get_label_to_indices()
        test_indices = [np.array(class_to_indices[clas]) for clas in classes_query]
        test_indices = np.concatenate(test_indices).tolist()        
        base_sampler = SequentialSampler(test_indices)
        test_sampler = BatchSampler(base_sampler, batch_size=32, drop_last=False)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, shuffle=False, batch_sampler=test_sampler,
            num_workers=C.n_worker, pin_memory=C.pin_memory,
        )
        # Dataloader for extended gallery
        eg_indices = list(range(len(self.dataset_eg)))   
        base_sampler = SequentialSampler(eg_indices)
        test_sampler = BatchSampler(base_sampler, batch_size=32, drop_last=False)
        self.eg_loader = torch.utils.data.DataLoader(
            self.dataset_eg, shuffle=False, batch_sampler=test_sampler,
            num_workers=C.n_worker, pin_memory=C.pin_memory,
        )
        

    def _get_dataset(self):
        train_df = pd.read_csv(C.train_meta_file, sep='@@@', header=None, engine='python')
        train_dataset = MUoMSGFSDataset(train_df)
        test_df = pd.read_csv(C.test_meta_file, sep='@@@', header=None, engine='python')
        test_dataset = UoMSGFSDataset(test_df)
        eg_df = pd.read_csv(C.eg_meta_file, sep='@@@', header=None, engine='python')
        eg_dataset = UoMSGFSDataset(eg_df)
        
        return train_dataset, test_dataset, eg_dataset

    def _get_dataset_expt(self):
        raise NotImplementedError

    def _get_weighted_random_sampler(self, dataset, weight_column):
        """
        Args:
            dataset (torch.utils.data.Dataset):
            weight_column (str):

        Returns:

        """
        raise NotImplementedError


class MVMLoader(object):
    """
    MVM data loader firstly splits all samples into train&validation 
    subset and test subset so that there is no overlapping identity among the two 
    subsets. It further splits the train&validation subset into train and validation
    subsets. All samples of an identity are exclusively allocated to either subset
    to ensure no overlapping identities exist among the subsets. This loader only 
    supports training with triplet or ranking losses, not applicable to that 
    requiring face identification.
    """
    def __init__(self):
        self.dataset, self.test_dataset, self.dataset_eg, self.valid_acc_dataset, self.cons_check_dataset = self._get_dataset()
        num_class= self.dataset.n_class()    # Total number of identities in self.dataset
        classes = self.dataset.label_set
        np.random.shuffle(classes)
        train_valid_percent = 1.0
        if not C.fixed_testset:
            train_valid_percent = 1.0 - C.test_set_size
        split = int(np.floor(train_valid_percent * num_class))
        train_valid_cls, test_cls = classes[:split], classes[split:]
        train_percent = 1 - C.valid_set_size
        split = int(np.floor(train_percent * len(train_valid_cls)))
        train_cls, valid_cls = train_valid_cls[:split], train_valid_cls[split:]
        
        train_sampler = MUoMSGFSBatchSampler(self.dataset.labels, #self.dataset.labels=array([[0,0,5],[0,1,0],...])
                                            self.dataset.label_to_indices, #self.dataset.label_to_indices={0:array([0,1]),1:array([2,3]),....}
                                            train_cls, C.n_bclass, C.n_sample_per_cls)
        valid_sampler = MUoMSGFSBatchSampler(self.dataset.labels,
                                            self.dataset.label_to_indices,
                                            valid_cls, C.n_bclass, C.n_sample_per_cls)
        
        # Dataloader for training dataset
        self.train_loader = torch.utils.data.DataLoader(
            self.dataset, batch_sampler=train_sampler, shuffle=False,
            num_workers=C.n_worker, pin_memory=C.pin_memory
        )
        self.valid_loader = torch.utils.data.DataLoader(
            self.dataset, batch_sampler=valid_sampler,
            num_workers=C.n_worker, pin_memory=C.pin_memory
        )
 
#        # Dataloader for train dataset # Temporarily used for Adapting Grad-CAM visualization method
#        classes_query1 = set(self.dataset.label_set)
#        class_to_indices1 = self.dataset.get_label_to_indices()
#        train_indices = [np.array(class_to_indices1[clas]) for clas in classes_query1]
#        train_indices = np.concatenate(train_indices).tolist()        
#        base_sampler = SequentialSampler(train_indices)
#        train_sampler = BatchSampler(base_sampler, batch_size=64, drop_last=False)
#        self.train_loader = torch.utils.data.DataLoader(
#            self.dataset, shuffle=False, batch_sampler=train_sampler,
#            num_workers=C.n_worker, pin_memory=C.pin_memory,
#        )
        # Dataloader for test dataset
        classes_query = set(self.test_dataset.label_set)
        if test_cls:
            old_test_cls = test_cls
            if C.reset_label:
                old_test_cls = [self.dataset.label_map[c] for c in test_cls]  # To original labels
            classes_query = list(classes_query.intersection(set(old_test_cls)))
        class_to_indices = self.test_dataset.get_label_to_indices()
        
        test_indices = [np.array(class_to_indices[clas]) for clas in classes_query]  # setA, setB and photo
#        test_indices = [[class_to_indices[clas][0], class_to_indices[clas][-1]] for clas in classes_query] # To extract the FileID and the ratings at the end of the meta data file 
        test_indices = np.concatenate(test_indices).tolist()        
        base_sampler = SequentialSampler(test_indices)
        test_sampler = BatchSampler(base_sampler, batch_size=32, drop_last=False)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, shuffle=False, batch_sampler=test_sampler,
            num_workers=C.n_worker, pin_memory=C.pin_memory,
        )
        # Dataloader for test extended gallery
        eg_indices = list(range(len(self.dataset_eg)))   
        base_sampler = SequentialSampler(eg_indices)
        test_sampler = BatchSampler(base_sampler, batch_size=32, drop_last=False)
        self.eg_loader = torch.utils.data.DataLoader(
            self.dataset_eg, shuffle=False, batch_sampler=test_sampler,
            num_workers=C.n_worker, pin_memory=C.pin_memory,
        )
        
        # Dataloader for calculating validation accuracy  
        # The labels must be converted back if they have been reset to integers           
        if C.valid_reset_label:
            valid_cls = [self.dataset.label_map[c] for c in valid_cls]  # To original labels
            valid_cls = [self.valid_acc_dataset.label_imap[c] for c in valid_cls]   # Convert original labels to internal labels
        class_to_indices = self.valid_acc_dataset.get_label_to_indices()
       
        valid_indices = [class_to_indices[clas][[0,-1]] for clas in valid_cls] # Get first sketch and photo of each subject
        valid_indices = np.concatenate(valid_indices).tolist()        
        base_sampler = SequentialSampler(valid_indices)
        valid_acc_sampler = BatchSampler(base_sampler, batch_size=32, drop_last=False)
        # valid_acc_sampler = VMSequentialSampler(self.valid_acc_dataset.labels, 
        #                                         self.valid_acc_dataset.label_to_indices,
        #                                         valid_cls, C.n_bclass)
        self.valid_acc_loader = torch.utils.data.DataLoader(
            self.valid_acc_dataset, shuffle=False, batch_sampler=valid_acc_sampler,
            num_workers=C.n_worker, pin_memory=C.pin_memory
        )
        
        # Dataloader for mismatch consistency check
        cons_check_indices=list(range(len(self.cons_check_dataset)))
        base_sampler =SequentialSampler(cons_check_indices)
        cons_check_sampler = BatchSampler(base_sampler, batch_size=30, drop_last=False)
        self.cons_check_loader = torch.utils.data.DataLoader(
            self.cons_check_dataset, shuffle=False, batch_sampler=cons_check_sampler,
            num_workers=C.n_worker, pin_memory=C.pin_memory,
        )
        
    def _get_dataset(self):
        if (C.sketch_type=='all'):
            # # To include both normal and occluded sketches
            train_df = pd.read_csv(C.train_meta_file, sep='@@@', header=None, engine='python')
            train_dataset = MVMDataset(train_df)
        elif (C.sketch_type=='occluded'):
            # To include only occluded sketches (drop normal sketches)
            train_df = pd.read_csv(C.train_meta_file, sep='@@@', header=None, engine='python')
            all_images=train_df.iloc[:,0].tolist()
            normal_sketches=[lis for lis in all_images if 'Org_Norm_VM_Set' in lis]
            for fn in normal_sketches:
                train_df.drop(train_df[train_df[0] == fn].index, inplace=True)
        elif (C.sketch_type=='normal'):
            # To include only normal sketches (drop occluded sketches)
            train_df = pd.read_csv(C.train_meta_file, sep='@@@', header=None, engine='python')
            all_images=train_df.iloc[:,0].tolist()
            occ_sketches=[lis for lis in all_images if 'AugComposites_selected' in lis]
            for fn in occ_sketches:
                train_df.drop(train_df[train_df[0] == fn].index, inplace=True)
        
        train_dataset = MVMDataset(train_df)
        print("Length of train_dataset.label_set is {0}".format(len(train_dataset.label_set)))
        print("Length of train_dataset.label_to_indices is {0}".format(len(train_dataset.get_label_to_indices())))
        test_df = pd.read_csv(C.test_meta_file, sep='@@@', header=None, engine='python')
        test_dataset = UoMSGFSDataset(test_df)
#        test_dataset = MVMDataset(test_df)  # For evaluating the automatic rating prediction (Need to have GT labels in the testset)
        # valid_acc_df=pd.read_csv(C.valid_acc_meta_file, sep='@@@', header=None, engine='python')
        # valid_acc_dataset=MVMDataset(valid_acc_df)
        valid_acc_dataset=MVMDataset(train_df) # Validate on the same training dataset
        
        cons_check_df=pd.read_csv(C.mismatch_cons_check_meta_file, sep='@@@', header=None, engine='python')
        # To drop the files which don't exist
        files=cons_check_df.iloc[:,0].tolist()
        clas=cons_check_df.iloc[:,1].tolist()
        i=0
        for cl, fn in zip(clas,files):
            if not os.path.exists(fn):
                i+=1
                print('{0} Dropped {1}'.format(i, cl))
                cons_check_df.drop(cons_check_df[cons_check_df[1]==cl].index,inplace=True)
        cons_check_dataset=MVMDataset(cons_check_df, order_rating=False) # For testing the model
        # cons_check_dataset=MVMDataset(cons_check_df, order_rating=True)    ########  Only for Consistency check 
        eg_df = pd.read_csv(C.eg_meta_file, sep='@@@', header=None, engine='python')
        eg_dataset = UoMSGFSDataset(eg_df)
        
        return train_dataset, test_dataset, eg_dataset, valid_acc_dataset, cons_check_dataset
        
    def _get_dataset_expt(self):
        raise NotImplementedError

    def _get_weighted_random_sampler(self, dataset, weight_column):
        """
        Args:
            dataset (torch.utils.data.Dataset):
            weight_column (str):

        Returns:

        """
        raise NotImplementedError
        

class MUoMSGFSLoader(object):
    """
    Morphed UoMSGFS data loader firstly splits all samples into train&valididation 
    subset and test subset so that there is no overlapping identity among the two 
    subsets. It further splits the train&validation subset into train and validation
    subsets. All samples of an identity are exclusively allocated to either subset
    to ENSURE NO OVERLAPPING IDENTITIES EXIST AMONG THE SUBSETS. This loader only 
    supports training with triplet or ranking losses, not applicable to that 
    requiring face identification.
    """
    def __init__(self):
        self.dataset, self.test_dataset, self.dataset_eg, self.valid_acc_dataset = self._get_dataset()
        num_class= self.dataset.n_class()    # 467
        classes = self.dataset.label_set
        np.random.shuffle(classes)
        train_valid_percent = 1.0
        if not C.fixed_testset:
            train_valid_percent = 1.0 - C.test_set_size
        split = int(np.floor(train_valid_percent * num_class))
        train_valid_cls, test_cls = classes[:split], classes[split:]
        train_percent = 1 - C.valid_set_size
        split = int(np.floor(train_percent * len(train_valid_cls)))
        train_cls, valid_cls = train_valid_cls[:split], train_valid_cls[split:]
        
        train_sampler = MUoMSGFSBatchSampler(self.dataset.labels, 
                                            self.dataset.label_to_indices,
                                            train_cls, C.n_bclass, C.n_sample_per_cls)
        valid_sampler = MUoMSGFSBatchSampler(self.dataset.labels,
                                            self.dataset.label_to_indices,
                                            valid_cls, C.n_bclass, C.n_sample_per_cls)
        # Dataloader for training dataset
        self.train_loader = torch.utils.data.DataLoader(
            self.dataset, batch_sampler=train_sampler, shuffle=False,
            num_workers=C.n_worker, pin_memory=C.pin_memory
        )
        self.valid_loader = torch.utils.data.DataLoader(
            self.dataset, batch_sampler=valid_sampler,
            num_workers=C.n_worker, pin_memory=C.pin_memory
        )
        # Dataloader for test dataset
        classes_query = set(self.test_dataset.label_set)
        if test_cls:
            old_test_cls = test_cls
            if C.reset_label:
                old_test_cls = [self.dataset.label_map[c] for c in test_cls]
            classes_query = list(classes_query.intersection(set(old_test_cls)))
        class_to_indices = self.test_dataset.get_label_to_indices()
        test_indices = [np.array(class_to_indices[clas]) for clas in classes_query]
        test_indices = np.concatenate(test_indices).tolist()        
        base_sampler = SequentialSampler(test_indices)
        test_sampler = BatchSampler(base_sampler, batch_size=32, drop_last=False)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, shuffle=False, batch_sampler=test_sampler,
            num_workers=C.n_worker, pin_memory=C.pin_memory,
        )
        # Dataloader for extended gallery
        eg_indices = list(range(len(self.dataset_eg)))   
        base_sampler = SequentialSampler(eg_indices)
        test_sampler = BatchSampler(base_sampler, batch_size=32, drop_last=False)
        self.eg_loader = torch.utils.data.DataLoader(
            self.dataset_eg, shuffle=False, batch_sampler=test_sampler,
            num_workers=C.n_worker, pin_memory=C.pin_memory,
        )
        
        #**************************Updated on Sep 21st 2021***********************************************
        # Dataloader for calculating validation accuracy  
        # The labels must be converted back if they have been reset to integers           
        if C.valid_reset_label:
            valid_cls = [self.dataset.label_map[c] for c in valid_cls]  # To original labels
            valid_cls = [self.valid_acc_dataset.label_imap[c] for c in valid_cls]   # Convert original labels to internal labels
        class_to_indices = self.valid_acc_dataset.get_label_to_indices()
       
        valid_indices = [class_to_indices[clas][[0,-1]] for clas in valid_cls] # Get first sketch and photo of each subject
        valid_indices = np.concatenate(valid_indices).tolist()        
        base_sampler = SequentialSampler(valid_indices)
        valid_acc_sampler = BatchSampler(base_sampler, batch_size=32, drop_last=False)
        # valid_acc_sampler = VMSequentialSampler(self.valid_acc_dataset.labels, 
        #                                         self.valid_acc_dataset.label_to_indices,
        #                                         valid_cls, C.n_bclass)
        self.valid_acc_loader = torch.utils.data.DataLoader(
            self.valid_acc_dataset, shuffle=False, batch_sampler=valid_acc_sampler,
            num_workers=C.n_worker, pin_memory=C.pin_memory
        )
        #**************************Updated on Sep 21st***********************************************

    def _get_dataset(self):
        train_df = pd.read_csv(C.train_meta_file, sep='@@@', header=None, engine='python')
        train_dataset = MUoMSGFSDataset(train_df)
        test_df = pd.read_csv(C.test_meta_file, sep='@@@', header=None, engine='python')
        test_dataset = UoMSGFSDataset(test_df)
        
        valid_acc_dataset=MUoMSGFSDataset(train_df) # Validate on the same training dataset
        
        eg_df = pd.read_csv(C.eg_meta_file, sep='@@@', header=None, engine='python')
        eg_dataset = UoMSGFSDataset(eg_df)
        
        return train_dataset, test_dataset, eg_dataset, valid_acc_dataset

    def _get_dataset_expt(self):
        raise NotImplementedError

    def _get_weighted_random_sampler(self, dataset, weight_column):
        """
        Args:
            dataset (torch.utils.data.Dataset):
            weight_column (str):

        Returns:

        """
        raise NotImplementedError
        
        
class SMUoMSGFSLoader_CV(object):
    """
    Stratified morphed UoMSGFS Data loader for cross validation
    """
    def __init__(self, n_fold):
        kf = KFold(n_splits=n_fold)
        self.dataset, self.test_dataset, self.dataset_eg = self._get_dataset()
        self.classes = np.array(self.dataset.label_set)
        np.random.shuffle(self.classes)        
        self.splits = kf.split(self.classes)
        self.n_fold = n_fold
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.eg_loader = None
        
    def __iter__(self):
        
        for i, (train_valid_cls_idx, test_cls_idx) in enumerate(self.splits):        
            # Split the sample list of very subject to train and valid parts. The composites and photos
            # must be split separately.
            train_valid_cls = self.classes[train_valid_cls_idx]
            test_cls = self.classes[test_cls_idx]
            labels = self.dataset.get_labels()
            ltoi = self.dataset.get_label_to_indices()
            ltoi = {cl:[ltoi[cl][labels[ltoi[cl],1]==0], ltoi[cl][labels[ltoi[cl],1]==1]] for cl in train_valid_cls}
            for cl in train_valid_cls:
                np.random.shuffle(ltoi[cl][0])
                np.random.shuffle(ltoi[cl][1])
            train_ltoi = {}
            valid_ltoi = {}   
            
            for cl in train_valid_cls:
                train_size_1 = math.floor(len(ltoi[cl][0]) * (1 - C.valid_set_size))
                train_size_2 = math.floor(len(ltoi[cl][1]) * (1 - C.valid_set_size))
                train_ltoi[cl] = [ltoi[cl][0][:train_size_1], ltoi[cl][1][:train_size_2]]
                valid_ltoi[cl] = [ltoi[cl][0][train_size_1:], ltoi[cl][1][train_size_2:]]
                
            train_ltoi = {cl:np.concatenate(train_ltoi[cl]) for cl in train_valid_cls}
            valid_ltoi = {cl:np.concatenate(valid_ltoi[cl]) for cl in train_valid_cls}
            
            train_sampler = MUoMSGFSBatchSampler(labels, 
                                                train_ltoi,
                                                train_valid_cls, C.n_bclass, C.n_sample_per_cls)
            valid_sampler = MUoMSGFSBatchSampler(labels,
                                                valid_ltoi,
                                                train_valid_cls, C.n_bclass, C.n_sample_per_cls)
            # Dataloader for training dataset
            self.train_loader = torch.utils.data.DataLoader(
                self.dataset, batch_sampler=train_sampler, shuffle=False,
                num_workers=C.n_worker, pin_memory=C.pin_memory
            )
            self.valid_loader = torch.utils.data.DataLoader(
                self.dataset, batch_sampler=valid_sampler,
                num_workers=C.n_worker, pin_memory=C.pin_memory
            )
            # Dataloader for test dataset
            classes_query = set(self.test_dataset.label_set)
            old_test_cls = [self.dataset.label_map[c] for c in test_cls]
            classes_query = list(classes_query.intersection(set(old_test_cls)))
            class_to_indices = self.test_dataset.get_label_to_indices()
            test_indices = [np.array(class_to_indices[clas]) for clas in classes_query]
            test_indices = np.concatenate(test_indices).tolist()        
            base_sampler = SequentialSampler(test_indices)
            test_sampler = BatchSampler(base_sampler, batch_size=32, drop_last=False)
            self.test_loader = torch.utils.data.DataLoader(
                self.test_dataset, shuffle=False, batch_sampler=test_sampler,
                num_workers=C.n_worker, pin_memory=C.pin_memory,
            )
            # Dataloader for extended gallery
            eg_indices = list(range(len(self.dataset_eg)))   
            base_sampler = SequentialSampler(eg_indices)
            test_sampler = BatchSampler(base_sampler, batch_size=32, drop_last=False)
            self.eg_loader = torch.utils.data.DataLoader(
                self.dataset_eg, shuffle=False, batch_sampler=test_sampler,
                num_workers=C.n_worker, pin_memory=C.pin_memory,
            )
        
            yield self
            
    def __len__(self):
        return self.n_fold

    def _get_dataset(self):
        train_df = pd.read_csv(C.train_meta_file, sep='@@@', header=None, engine='python')
        train_dataset = MUoMSGFSDataset(train_df)
        test_df = pd.read_csv(C.test_meta_file, sep='@@@', header=None, engine='python')
        test_dataset = UoMSGFSDataset(test_df)
        eg_df = pd.read_csv(C.eg_meta_file, sep='@@@', header=None, engine='python')
        eg_dataset = UoMSGFSDataset(eg_df)
        
        return train_dataset, test_dataset, eg_dataset


class MUoMSGFSLoader_CV(object):
    """
    Morphed UoMSGFS Data loader for cross validation
    """
    def __init__(self, n_fold):
        kf = KFold(n_splits=n_fold)
        self.dataset, self.test_dataset, self.dataset_eg = self._get_dataset()
        self.classes = np.array(self.dataset.label_set)
        np.random.shuffle(self.classes)        
        self.splits = kf.split(self.classes)
        self.n_fold = n_fold
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.eg_loader = None
        
    def __iter__(self):
        
        for i, (train_valid_cls_idx, test_cls_idx) in enumerate(self.splits):        
            # Split the sample list of very subject to train and valid parts. The composites and photos
            # must be split separately.
            train_valid_cls = self.classes[train_valid_cls_idx]
            test_cls = self.classes[test_cls_idx]
            train_percent = 1 - C.valid_set_size
            split = int(np.floor(train_percent * len(train_valid_cls)))
            train_cls, valid_cls = train_valid_cls[:split], train_valid_cls[split:]
            
            train_sampler = MUoMSGFSBatchSampler(self.dataset.labels, 
                                                self.dataset.label_to_indices,
                                                train_cls, C.n_bclass, C.n_sample_per_cls)
            valid_sampler = MUoMSGFSBatchSampler(self.dataset.labels,
                                                self.dataset.label_to_indices,
                                                valid_cls, C.n_bclass, C.n_sample_per_cls)
            # Dataloader for training dataset
            self.train_loader = torch.utils.data.DataLoader(
                self.dataset, batch_sampler=train_sampler, shuffle=False,
                num_workers=C.n_worker, pin_memory=C.pin_memory
            )
            self.valid_loader = torch.utils.data.DataLoader(
                self.dataset, batch_sampler=valid_sampler,
                num_workers=C.n_worker, pin_memory=C.pin_memory
            )
            # Dataloader for test dataset
            classes_query = set(self.test_dataset.label_set)
            old_test_cls = [self.dataset.label_map[c] for c in test_cls]
            classes_query = list(classes_query.intersection(set(old_test_cls)))
            class_to_indices = self.test_dataset.get_label_to_indices()
            test_indices = [np.array(class_to_indices[clas]) for clas in classes_query]
            test_indices = np.concatenate(test_indices).tolist()        
            base_sampler = SequentialSampler(test_indices)
            test_sampler = BatchSampler(base_sampler, batch_size=32, drop_last=False)
            self.test_loader = torch.utils.data.DataLoader(
                self.test_dataset, shuffle=False, batch_sampler=test_sampler,
                num_workers=C.n_worker, pin_memory=C.pin_memory,
            )
            # Dataloader for extended gallery
            eg_indices = list(range(len(self.dataset_eg)))   
            base_sampler = SequentialSampler(eg_indices)
            test_sampler = BatchSampler(base_sampler, batch_size=32, drop_last=False)
            self.eg_loader = torch.utils.data.DataLoader(
                self.dataset_eg, shuffle=False, batch_sampler=test_sampler,
                num_workers=C.n_worker, pin_memory=C.pin_memory,
            )
        
            yield self
            
    def __len__(self):
        return self.n_fold

    def _get_dataset(self):
        train_df = pd.read_csv(C.train_meta_file, sep='@@@', header=None, engine='python')
        train_dataset = MUoMSGFSDataset(train_df)
        test_df = pd.read_csv(C.test_meta_file, sep='@@@', header=None, engine='python')
        test_dataset = UoMSGFSDataset(test_df)
        eg_df = pd.read_csv(C.eg_meta_file, sep='@@@', header=None, engine='python')
        eg_dataset = UoMSGFSDataset(eg_df)
        
        return train_dataset, test_dataset, eg_dataset
    

class MUoMSGFSALoader(MUoMSGFSLoader):
    """
    Morphed UoMSGFS data loader firstly splits all samples into train&valididation 
    subset and test subset so that there is no overlapping identity among the two 
    subsets. It further splits the train&validation subset into train and validation
    subsets. All samples of an identity are exclusively allocated to either subset
    to ensure no overlapping identities exist among the subsets. This loader only 
    supports training with triplet or ranking losses, not applicable to that 
    requiring face identification.
    """
    def __init__(self):
        super(MUoMSGFSALoader, self).__init__()

    def _get_dataset(self):
        train_df = pd.read_csv(C.train_meta_file, sep='@@@', header=None, engine='python')
        train_attr_df = pd.read_csv(C.attr_meta_file, sep=',', header=None, engine='python')
        train_dataset = MUoMSGFSADataset(train_df, train_attr_df)
        test_df = pd.read_csv(C.test_meta_file, sep='@@@', header=None, engine='python')
        test_dataset = UoMSGFSDataset(test_df)
        eg_df = pd.read_csv(C.eg_meta_file, sep='@@@', header=None, engine='python')
        eg_dataset = UoMSGFSDataset(eg_df)
        
        return train_dataset, test_dataset, eg_dataset


class SMUoMSGFSALoader(SMUoMSGFSLoader):
    """
    Morphed UoMSGFS data loader firstly splits all samples into train&valididation 
    subset and test subset so that there is no overlapping identity among the two 
    subsets. It further splits the train&validation subset into train and validation
    subsets. All samples of an identity are exclusively allocated to either subset
    to ensure no overlapping identities exist among the subsets. This loader only 
    supports training with triplet or ranking losses, not applicable to that 
    requiring face identification.
    """
    def __init__(self):
        super(SMUoMSGFSALoader, self).__init__()

    def _get_dataset(self):
        train_df = pd.read_csv(C.train_meta_file, sep='@@@', header=None, engine='python')
        train_attr_df = pd.read_csv(C.attr_meta_file, sep=',', header=None, engine='python')
        train_dataset = MUoMSGFSADataset(train_df, train_attr_df)
        test_df = pd.read_csv(C.test_meta_file, sep='@@@', header=None, engine='python')
        test_dataset = UoMSGFSDataset(test_df)
        eg_df = pd.read_csv(C.eg_meta_file, sep='@@@', header=None, engine='python')
        eg_dataset = UoMSGFSDataset(eg_df)
        
        return train_dataset, test_dataset, eg_dataset
    

class MUoMSGFSALoader_CV(MUoMSGFSLoader_CV):
    """
    Morphed UoMSGFSA Data loader for cross validation
    """
    def __init__(self, n_fold):
        super(MUoMSGFSALoader_CV, self).__init__(n_fold)
        
    def _get_dataset(self):
        train_df = pd.read_csv(C.train_meta_file, sep='@@@', header=None, engine='python')
        attr_df = pd.read_csv(C.attr_meta_file, sep='@@@', header=None, engine='python')
        train_dataset = MUoMSGFSADataset(train_df, attr_df)
        test_df = pd.read_csv(C.test_meta_file, sep='@@@', header=None, engine='python')
        test_dataset = UoMSGFSDataset(test_df)
        eg_df = pd.read_csv(C.eg_meta_file, sep='@@@', header=None, engine='python')
        eg_dataset = UoMSGFSDataset(eg_df)
        
        return train_dataset, test_dataset, eg_dataset


class SMUoMSGFSALoader_CV(SMUoMSGFSLoader_CV):
    """
    Stratified Morphed UoMSGFSA Data loader for cross validation
    """
    def __init__(self, n_fold):
        super(SMUoMSGFSALoader_CV, self).__init__(n_fold)
        
    def _get_dataset(self):
        train_df = pd.read_csv(C.train_meta_file, sep='@@@', header=None, engine='python')
        attr_df = pd.read_csv(C.attr_meta_file, sep='@@@', header=None, engine='python')
        train_dataset = MUoMSGFSADataset(train_df, attr_df)
        test_df = pd.read_csv(C.test_meta_file, sep='@@@', header=None, engine='python')
        test_dataset = UoMSGFSDataset(test_df)
        eg_df = pd.read_csv(C.eg_meta_file, sep='@@@', header=None, engine='python')
        eg_dataset = UoMSGFSDataset(eg_df)
        
        return train_dataset, test_dataset, eg_dataset
    
# Main function goes here    
if __name__ == "__main__":
    C.reset_label = False
    data_frame = pd.read_csv(C.train_meta_file, sep='@@@', header=None, engine='python')
    dataset = MUoMSGFSDataset(data_frame)    
#    fig = plt.figure()
#    
#    for i in range(len(dataset)):
#        x,y = dataset[i]
#        #x = x.detach().cpu().numpy()
#        x = np.array(x)
#        print(i, x.shape, y.shape, y)
#        plt.tight_layout()
#        x = x / np.max(x)
#        plt.imshow(x)
#        plt.show()
#        plt.pause(1)        
#
#        if i==3:
#            break

    sampler = None
    if dataset.get_labels() is not None:
        sampler = MUoMSGFSBatchSampler(dataset.get_labels(), dataset.get_label_to_indices(), dataset.label_set, 5)
        print(len(sampler)) # 229,920 triplets // (5*4) = 11,496
    
        loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, sampler=sampler,
        num_workers=0, pin_memory=False
    )
    
    for batch_idx, data in enumerate(loader):
        print(batch_idx)
        print(data[0].shape)
        
        
    