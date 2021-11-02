from itertools import combinations, permutations
import numpy as np
import torch
import os
import sys
import six
import shutil
from sklearn import manifold
#from config import Configuration as C
from tqdm import tqdm
from sklearn.metrics import pairwise_distances, pairwise
import pandas as pd
import xlsxwriter


C = None

def set_config(_C):
    global C
    C = _C
    
#def initialize_training_dir():
#    paths = [C.train_summary_dir,
#             C.valid_summary_dir,
#             C.test_summary_dir,
#             C.trained_model_dir,
#             ]
#    for path in paths:
#        if not os.path.exists(path):
#            os.makedirs(path)
#        else:
#            if not int(input('The path [' + path.resolve().name + '] exist. Keep it? (1 or 0) --> ')):
#                shutil.rmtree(path)
#                os.makedirs(path)

def embed2tsne(emd, label):
    emd = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    gender_lbl = np.array([1, 1, 0, 0])
    race_lbl = np.array([0, 1, 0, 2])
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    x_tsne = tsne.fit_transform(emd)
    print("Original data dimension is {}. Embedded data dimension is {}".format(emd.shape[-1], x_tsne.shape[-1]))
    x_min, x_max = x_tsne.min(0), x_tsne.max(0)
    x_norm = (x_tsne - x_min) / (x_max - x_min)


def load_model(model_path, model_name):
    """Load imoprted PyTorch model by name

    Args:
        model_name (str): the name of the model to be loaded

    Return:
        nn.Module: the loaded network
    """
    model_def_path = os.path.join(model_path, model_name + ".py")
    weights_path = os.path.join(model_path, model_name + ".pth")
    if six.PY3:
        import importlib.util

        spec = importlib.util.spec_from_file_location(model_name,
                                                      model_def_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        import importlib
        dirname = os.path.dirname(model_def_path)
        sys.path.insert(0, dirname)
        module_name = os.path.splitext(os.path.basename(model_def_path))[0]
        mod = importlib.import_module(module_name)
    func = getattr(mod, model_name)
    net = func(weights_path=weights_path)
    return net   


def fix_parameters(module, fix):
    for param in module.parameters():
        param.requires_grad = not fix
        
def load_module_2or3(model_name, model_def_path):
    """Load model definition module in a manner that is compatible with
    both Python2 and Python3

    Args:
        model_name: The name of the model to be loaded
        model_def_path: The filepath of the module containing the definition

    Return:
        The loaded python module."""
    if six.PY3:
        import importlib.util
        spec = importlib.util.spec_from_file_location(model_name, model_def_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        import importlib
        dirname = os.path.dirname(model_def_path)
        sys.path.insert(0, dirname)
        module_name = os.path.splitext(os.path.basename(model_def_path))[0]
        mod = importlib.import_module(module_name)
    return mod


def pdist(vectors, metric='euclidean'):
    if metric=='euclidean':
        distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
                dim=1).view(-1, 1)  #-2ab+a^2+b^2
        distance_matrix = distance_matrix.sqrt()
    elif metric=='cosine':
        # It is assumed that vectors have been normalized to have unit norms
        distance_matrix = 1 - vectors.mm(torch.t(vectors))
        distance_matrix = torch.relu(distance_matrix)
    return distance_matrix
def psim(vectors, metric='cosine_sim'):
    if metric=='cosine_sim':
        # It is assumed that vectors have been normalized to have unit norms
        sim_matrix = vectors.mm(torch.t(vectors))
         # relu is removed here, because similarity could be negative
    return sim_matrix

def getFileName(filename):
    img_path=filename
    img_path=img_path.strip('\n')
    img_filename = os.path.split(img_path)[-1]
    img_filename =os.path.splitext(img_filename)[0]
    return img_filename

def getFileId(filename):
    img_path=filename
    img_path=img_path.strip('\n')
    img_filename = os.path.split(img_path)[-1]
    img_filename =os.path.splitext(img_filename)[0]
    fileid=img_filename.split("_")[0]
    if fileid.isnumeric():
        fileid = int(fileid)
    return fileid

def getFileId_VM(filename):
    img_path=filename
    img_path=img_path.strip('\n')
    img_filename = os.path.split(img_path)[-1]
    img_filename_f =os.path.splitext(img_filename)[0]
    # # 44 Real EFITs
    # if "_BM_" in img_filename_f:
    #     fileid=img_filename_f
        
    if "_BM_" in img_filename_f:
        fileid=img_filename.split("_BM_")[1]
        fileid=fileid.split(".")[0]
    elif "_BM_1_" in img_filename_f:
        fileid=img_filename.split("_BM_1_")[1]
        fileid=fileid.split(".")[0]
    elif "_PI_1_" in img_filename_f:
        fileid=img_filename.split("_PI_1_")[1]
        fileid=fileid.split(".")[0]
    elif "_PI_2_" in img_filename_f:
        fileid=img_filename.split("_PI_2_")[1]
        fileid=fileid.split(".")[0]
    elif "_PI_3_" in img_filename_f:
        fileid=img_filename.split("_PI_3_")[1]
        fileid=fileid.split(".")[0]
    elif "_PI_4_" in img_filename_f:
        fileid=img_filename.split("_PI_4_")[1]
        fileid=fileid.split(".")[0]
    elif "_MIS_1_" in img_filename_f:
        fileid=img_filename.split("_MIS_1_")[1]
        fileid=fileid.split(".")[0]
    elif "_MIS_2_" in img_filename_f:
        fileid=img_filename.split("_MIS_2_")[1]
        fileid=fileid.split(".")[0]
    elif "_MIS_3_" in img_filename_f:
        fileid=img_filename.split("_MIS_3_")[1]
        fileid=fileid.split(".")[0]
    elif "_MIS_4_" in img_filename_f:
        fileid=img_filename.split("_MIS_4_")[1]
        fileid=fileid.split(".")[0]
    else:
        fileid=img_filename
        fileid=fileid.split(".")[0]
    # if fileid.isnumeric():
    #     fileid = int(fileid)
    return fileid
class EmbeddingComparator:
    """
    
    """
    def __init__(self):
        raise NotImplementedError
    
    def get_rank_list(self, query, gallery, top_n):
        raise NotImplementedError
        
    def get_metric_type(self):
        raise NotImplementedError

class EmbeddingComparator_L2(EmbeddingComparator):
    """
    """
    def __init__(self):
        return
    
    def get_rank_list(self, query, gallery, top_n):
        distances = pairwise_distances(query.reshape(1, -1), gallery, metric='euclidean')
        sort_dist=np.sort(distances)
        indices = np.argsort(distances)[0][:top_n] # argsort - Return the indices of an sorted array
        return sort_dist, indices

    
    def get_metric_type(self):
        return 'euclidean'
    
class EmbeddingComparator_Verif(EmbeddingComparator):
    """
    """
    def __init__(self, net):
        self.net = net
        
    def get_rank_list(self, query, gallery, top_n):
        if isinstance(query, np.ndarray):
            query = torch.from_numpy(query)
        if isinstance(gallery, np.ndarray):
            gallery = torch.from_numpy(gallery)
        if not isinstance(query, torch.Tensor) or not isinstance(gallery, torch.Tensor):
            raise TypeError
        
        prob = []
        x_q = query.unsqueeze(0)
        if next(self.net.parameters()).is_cuda:
            x_q = x_q.cuda()
            
        for x_g in gallery:
            x_g = x_g.unsqueeze(0)
            if next(self.net.parameters()).is_cuda:
                x_g = x_g.cuda()
            y = self.net.verify(x_q, x_g)
            prob += y.detach().cpu().numpy().tolist()
        
        indices = np.flip(np.argsort(prob))[:top_n]
        return indices        
    
    def get_metric_type(self):
        return 'verif'
    
class EmbeddingComparator_Cosine(EmbeddingComparator):
    """
    """
    def __init__(self):
        return
    
    def get_rank_list(self, query, gallery, top_n):
        distances = pairwise_distances(query.reshape(1, -1), gallery, metric='cosine')
        indices = np.argsort(distances)[0][:top_n]
        sort_dist=np.sort(distances)
        cos_sim=[]
        for dist in sort_dist:
            cos_sim.append(1-dist)
        return cos_sim, indices        
    
    def get_metric_type(self):
        return 'cosine'

def normalize_emd(emd: np.ndarray):
    centroid_vector = np.mean(emd, axis=0, keepdims=True)
    if len(emd) > 1:
        emd = emd - centroid_vector
    emd = emd / np.linalg.norm(emd, axis=-1, keepdims=True)
    return emd
        

        
def calculate_accuracy(embeddings_gallery, embeddings_query, ranks, percent_ranks, comparator,wb,sheet_name, row,col):
    '''
    Input:
        embeddings_gallery
        embeddings_query
        ranks
        comparator
    Return:
        accuracies: 
    '''
    i1=0
    i2=0
    sheet=wb.add_worksheet(str(sheet_name))

    
    #ranks = [1, 5, 10, 50, 100]
    hit=np.zeros((len(ranks),), dtype=np.float32)
    hit1=np.zeros((len(ranks),), dtype=np.float32)
    n_correct = np.zeros((len(ranks),), dtype=np.float32)
    n_correct1 = np.zeros((len(ranks),), dtype=np.float32)
    accuracies = np.zeros((len(ranks),), dtype=np.float32)
    cal_accuracies = np.zeros((len(percent_ranks),), dtype=np.float32)
    cal_ranks=np.zeros((len(percent_ranks),), dtype=np.int)
    overall_rank=0
    weights=C.weights
    for i in tqdm(range(len(embeddings_query))): 
        simscore, similar_indices = comparator.get_rank_list(embeddings_query[i], 
                                                   embeddings_gallery, 
                                                  top_n=max(ranks))
        
        for j in range(len(ranks)):
            if i in similar_indices[:ranks[j]]:
                n_correct[j]+=1
            hit[j]=n_correct[j]
            
        for jdx in range(len(percent_ranks)):
            cal_ranks[jdx]=len(embeddings_gallery)*(percent_ranks[jdx]/100)
            cal_ranks[jdx]=round(cal_ranks[jdx])
        simscore1, similar_indices1 = comparator.get_rank_list(embeddings_query[i], 
                                                   embeddings_gallery, 
                                                   top_n=max(cal_ranks))
        for jdx in range(len(percent_ranks)):
            if i in similar_indices1[:cal_ranks[jdx]]:
                n_correct1[jdx]+=1
            hit1[jdx]=n_correct1[jdx]
            
    sheet.write(row,0,"log_fname")    
    sheet.write(col+1,0,C.log_fname)
    sheet.write(row,1,"Config_fname".format(C.config_fname))    
    sheet.write(col+1,1,C.config_fname)
    for i in range(len(ranks)):
        accuracies[i] = n_correct[i] / len(embeddings_query)
        print ("\nRank-{0}-Accuracy {1:.2f}%-({2}-hit(s))".format(ranks[i], accuracies[i]*100, int(hit[i])))
        sheet.write(row,i+2,"Rank-{0}".format(ranks[i]))
        sheet.write(col+1,i+2,"{0:.2f}%".format(accuracies[i]*100))
        overall_rank+=accuracies[i]*100*weights[i]
    i1=i+3
    print ("\nOverall Rank Accuracy {1:.2f}%".format(1,overall_rank)) 
    sheet.write(row,i1,"Overall Rank")
    sheet.write(col+1,i1,"{0:.2f}%".format(overall_rank))
    i2=i1+1
    for ip in range(len(percent_ranks)):
        cal_accuracies[ip] = n_correct1[ip] / len(embeddings_query)
        print ("\nRank-{0}-{1}%-Accuracy {2:.2f}%-({3}-hit(s))".format(cal_ranks[ip], percent_ranks[ip], cal_accuracies[ip]*100, int(hit1[ip])))
        sheet.write(row,i2+ip,"Rank-{0}%".format(percent_ranks[ip]))
        sheet.write(col+1,i2+ip,"{0:.2f}%".format(cal_accuracies[ip]*100))
  
    return accuracies, overall_rank, cal_accuracies

def check_rank_hit(query_cls, cls_eg, ranked_indices, ranks):
    hit = np.zeros((len(ranks),), dtype=np.int)
    cls_eg = [cls_eg[ranked_indices[i]] for i in ranked_indices]
    
    for i,r in enumerate(ranks):
        if query_cls in cls_eg[:r]:
            hit[i] = 1
            
    return hit

#def get_embeddings(net, data_loader):
#    embeddings = []
#    net.to(C.device)
#    net.eval()
#    labels = []
#    
#    with torch.no_grad():
#        for batch_idx, data in enumerate(tqdm(data_loader)):
#            target = None
#            if type(data) in (list, tuple):
#                target = data[1]
#                data = data[0]
#                if type(data) in (list, tuple):
#                    data = data[0]
#            if target is not None: 
#                if type(target) == torch.Tensor:
#                    _labels = target.detach().numpy()
#                elif type(target) == tuple:
#                    _labels = list(target)
#                else:
#                    _labels = target
#                labels.append(_labels)
#            data = data.to(C.device)
#            embeddings_ = net.get_embedding(data)
#            embeddings_ = embeddings_.detach().cpu().numpy()
#            embeddings.append(embeddings_)
#    
#    embeddings = np.vstack(embeddings)
#    if labels: 
#        labels = np.concatenate(labels)
#    return embeddings, labels

def get_embeddings(net, data_loader, valid_reset_label=False):
    embeddings = []
    feature_map=[]
    
    net.to(C.device)
    net.eval()
    labels = []
    ratings=[]
    test_reset_label=True  ####
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(data_loader)):
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
                if np.ndim(_labels) > 1:  # if target contains label, modality and ratings
                    test_reset_label=False
                    # _ratings = _labels[:,2]
                    # ratings.append(_ratings)
                    _labels =_labels[:,0]
                #Convert integer labels back to original labels if necessary
                if (C.valid_reset_label and not test_reset_label):
                    _labels=[data_loader.dataset.label_map[c] for c in _labels]
                labels.append(_labels)
            # data = data.to(C.device) # Uncomment for general training apart from using 2 trained face parsers
            
            embeddings_, feature_map_ = net.get_embedding(data)  # feature_map_ (Added for visualization)
            # embeddings_= net.get_embedding(data)  # feature_map_ (Added for visualization)
            embeddings_=embeddings_.cpu()  # Convert CUDA tensor to CPU tensor
            feature_map_=feature_map_.cpu()  # Convert CUDA tensor to CPU tensor
            
            embeddings.append(embeddings_)
            feature_map.append(feature_map_)   # Added for visualization 
    
    embeddings = np.vstack(embeddings)
    feature_map = np.vstack(feature_map)  # Added for visualization 
    if labels: 
        labels = np.concatenate(labels)
    if ratings:
        ratings=np.concatenate(ratings)
    # return embeddings, labels
    return embeddings, feature_map, labels   # Added for visualization 
    