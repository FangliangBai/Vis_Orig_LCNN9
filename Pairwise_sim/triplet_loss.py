import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
from utils import pdist

C = None

def set_config(_C):
    global C
    C = _C
    
class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError


def hardest_negative(loss_values): # Hard negatives # Take the max loss
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None


def random_hard_negative(loss_values):  # Random Easy negatives 
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None

def all_hard_negative(loss_values):  # Easy negatives
    all_negatives = np.where(loss_values > 0)[0]
    return all_negatives if len(all_negatives) > 0 else None

def semihard_negative(loss_values, margin): # Semi-hard negatives
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    #return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None
    return semihard_negatives

#Softmax function
#def softmax(x,x_set):
##    e_x=np.exp(x-np.max(x)) # Range differs
#    e_x=np.exp(x)    
#    return e_x/x_set.sum()

def softmax(x,x_set):
    e_x=np.exp(x)  
    e_x_set=np.exp(x_set)
    den=e_x_set.sum()
    sm=e_x/den
    return sm

class FunctionNegativeTripletSelector_UoMSGFS(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector_UoMSGFS, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels, metric):
        """
        Input:
            embeddings: Embeddings of training samples in a batch, N x m where N is the number of samples
                        and m is the number of dimensions
            labels:     N x 1 or N x 2 Labels of training samples. If one column, it contains identity
                        IDs. In this case, it is assumed that each identity has equal numbers of 
                        composites and photos and the embeddings must be arranged as 
                        If two column, the second column is the modality of samples and there is no
                        special arrangement for embeddings
        """
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings, metric) # shape ([64,64])
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()  # len(labels)=64
        modalities = None

        if (np.ndim(labels) >= 2): # in case of extended labels where the second column is image modality
            labels, modalities = labels[:,0], labels[:,1] # Actual
            # labels, modalities, ratings = labels[:,0], labels[:,1], labels[:,2]  # Added by Siva
            # ratings_set = np.sort(np.unique(ratings).tolist())  
            # new_ratings_set=np.delete(ratings_set,0)
            
        triplets = []

        for anchor_label in set(labels):
            label_mask = (labels == anchor_label)
            label_indices = np.sort(np.where(label_mask)[0])
            if len(label_indices) < 2:
                continue
            
            #negative_indices = np.sort(np.where(np.logical_not(label_mask))[0])[1::2] # skip negative composites
            #anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            negative_indices = np.sort(np.where(np.logical_not(label_mask))[0]) 
            if (C.anchor_type == 'sketch_only'):
                # Anchors are of sketch type
                if modalities is not None:
                    negphoto_indices = negative_indices[modalities[negative_indices]==1]            
                    compo_list = label_indices[modalities[label_indices]==0].tolist()
                    photo_list = label_indices[modalities[label_indices]==1].tolist()
                else:
                    n_sample_per_mode = len(label_indices) // 2  # 1 # assuming Comp/photo/Comp/photo/....
                    negphoto_indices = negative_indices[negative_indices//n_sample_per_mode % 2 == 1]
                    compo_list = label_indices[:len(label_indices)//2].tolist() # first composite
                    photo_list = label_indices[len(label_indices)//2:].tolist() # next photo             
            elif (C.anchor_type == 'both'):
                if modalities is not None:
                    # # Anchors are of sketch type
                    negphoto_indices = negative_indices[modalities[negative_indices]==1] # Choose only negative photos
                    # compo_list = label_indices[modalities[label_indices]==0].tolist()
                    # photo_list = label_indices[modalities[label_indices]==1].tolist()
                    # # Anchors are of both sketch and photo type
                    compo_list=label_indices.tolist()
                    photo_list = label_indices.tolist()
                # else:
                #     n_sample_per_mode = len(label_indices) // 2  # 1 # assuming Comp/photo/Comp/photo/....
                #     negphoto_indices = negative_indices[negative_indices//n_sample_per_mode % 2 == 1]
                #     compo_list = label_indices[:len(label_indices)//2].tolist() # first composite
                #     photo_list = label_indices[len(label_indices)//2:].tolist() # next photo     
            if not compo_list or not photo_list:
                continue
            
            anchor_positives = []
            if (C.anchor_type == 'sketch_only'):
                # It might be possible that the number of composites is different from that of photos
                if len(compo_list) >= len(photo_list):     
                    # Anchors are of sketch type
                    for x in itertools.permutations(compo_list, len(photo_list)):
                        anchor_positives += list(zip(x, photo_list))
                    else:
                        for x in itertools.permutations(photo_list, len(compo_list)):
                            anchor_positives += list(zip(compo_list, x))          
            elif (C.anchor_type == 'both'):    
                # It might be possible that the number of composites is different from that of photos
                if len(compo_list) >= len(photo_list):     
                    # # Anchors are of sketch type
                    # for x in itertools.permutations(compo_list, len(photo_list)):
                    #     anchor_positives += list(zip(x, photo_list))
                    # Anchors are of both sketch and photo type
                    photo_list=np.array(photo_list)
                    for x in itertools.permutations(compo_list, len(photo_list)-1):
                        # print(x) 
                        for i in range(len(x)):
                            if (modalities[x[i]]==0):
                                dum_photo_list=photo_list[modalities[photo_list]==1].tolist()
                                anchor_positives += list(zip((x[i],), dum_photo_list))
                            elif (modalities[x[i]]==1):
                                dum_photo_list=photo_list[modalities[photo_list]==0].tolist()
                                anchor_positives += list(zip((x[i],), dum_photo_list))
                # else:
                #     for x in itertools.permutations(photo_list, len(compo_list)):
                #         anchor_positives += list(zip(compo_list, x))
                    
            anchor_positives = np.array(anchor_positives)            
            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]] # Update weights to Composites (Above line Doesn't make sense, hence ,ultiplied the ratings with loss values)
            
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                if (C.consider_ratings_during_train == False):
##********************************************************************************************************************************                
    #                # triplet loss without human ratings
    
                    loss_values = ap_distance - distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), 
                                            torch.LongTensor(negphoto_indices)] + self.margin # loss_values.shape = torch.Size([31])])
                elif (C.consider_ratings_during_train == True):
#********************************************************************************************************************************        
    ##              # triplet loss by incorporating human ratings
                    loss_values = ap_distance*softmax(ratings[anchor_positive[0]], new_ratings_set) - distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), 
                                            torch.LongTensor(negphoto_indices)] + self.margin # Exp-1
#********************************************************************************************************************************                
#                triplet loss = d(a,p)- d(a,n) + margin  # Actual formula
#                triplet loss = d(a,p)*softmax[alpha(BM:10, PI:5, MS:1)] - d(a,n) + margin - Exp-1
#                triplet loss = d(a,p) - d(a,n)*softmax[alpha(BM:1, PI:5, MS:10)] + margin - Exp-2
#                triplet loss = d(a,p)*softmax[alpha(BM:10, PI:5, MS:1)] - d(a,n)*softmax[alpha(BM:1, PI:5, MS:10)] + margin - Exp 3
                # Inter-class triplet loss: Intra-class difference
#                loss_values = loss_values * ratings[anchor_positive[0]]  # Multiply the loss values with the ratings of the anchor
#                loss_values = loss_values * np.exp(ratings[anchor_positive[0]])  # Multiply the loss values with the exponentiation of the ratings of the anchor
#********************************************************************************************************************************                
                loss_values = loss_values.data.cpu().numpy()
                hard_negatives = self.negative_selection_fn(loss_values) ## Hard negatives for which loss will be greater than zero 
                if hard_negatives is not None:
                    hard_negatives = negphoto_indices[hard_negatives]   # Choose all the qualifying negative indices
                    if type(hard_negatives) is not (list, tuple) and not isinstance(hard_negatives, np.ndarray):
                        hard_negatives = [hard_negatives]
                    # Add all selected triplets
                    for x in hard_negatives:
                        triplets.append([anchor_positive[0], anchor_positive[1], x])

        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negphoto_indices[0]]) # If there are no hard negatives, take the first negative 
 
        triplets = np.array(triplets)

        return torch.LongTensor(triplets)
        
def HardestNegativeTripletSelector(cpu=False): 
    return FunctionNegativeTripletSelector_UoMSGFS(margin=C.tri_loss_param['margin'],
                                           negative_selection_fn=hardest_negative,
                                           cpu=cpu)


def RandomNegativeTripletSelector(cpu=False): 
    return FunctionNegativeTripletSelector_UoMSGFS(margin=C.tri_loss_param['margin'],
                                           negative_selection_fn=random_hard_negative,
                                           cpu=cpu)


def SemihardNegativeTripletSelector(cpu=False): 
    return FunctionNegativeTripletSelector_UoMSGFS(margin=C.tri_loss_param['margin'],
                                           negative_selection_fn=lambda x: semihard_negative(x, C.tri_loss_param['margin']),
                                           cpu=cpu)


def AllNegativeTripletSelector(cpu=False): 
    return FunctionNegativeTripletSelector_UoMSGFS(margin=C.tri_loss_param['margin'],
                                           negative_selection_fn=all_hard_negative,
                                           cpu=cpu)

class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = C.tri_loss_param['margin']  # margin:1
        self.triplet_selector = triplet_selector  # triplet_selector:All/hard/semi-hard
        self.metric=C.tri_loss_param['metric']  # metric:Euclidean
        
    def forward(self, x, target):
        
        if type(x) in (tuple, list):
            x, embeddings = x
        else:
            embeddings = x
        
        if self.metric == 'cosine':
            embeddings = embeddings / (1e-10 + torch.norm(embeddings, dim=1, keepdim=True))
            
        triplets = self.triplet_selector.get_triplets(embeddings, target, self.metric)
        
        if embeddings.is_cuda:
            triplets = triplets.cuda()

        if self.metric=='euclidean':
            ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1).sqrt()  # .pow(.5)
            an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1).sqrt()  # .pow(.5)
        else:
            anchors = embeddings[triplets[:, 0]]
            pos = embeddings[triplets[:, 1]]
            neg = embeddings[triplets[:, 2]]
            ap_distances = 1.0 - (anchors * pos).sum(1)
            an_distances = 1.0 - (anchors * neg).sum(1)
        losses = F.relu(ap_distances - an_distances + self.margin)
#        losses = ap_distances - an_distances + self.margin  # For visualization, discard hinge function(Relu)
        
#        compact_loss = (torch.abs(embeddings) - torch.mean(embeddings, 1, keepdim=True)).pow(2).sum(1)
#        compact_loss = compact_loss.mean()       
#        norm_loss = embeddings.pow(2).sum(1).sqrt().mean()
        
        return losses.mean(), len(triplets)# the latter is for calculating metrics   # Training

   
        
        
class OnlineTripletLoss_vis(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, triplet_selector):
        self.gradients = []   # For Visualization
        super(OnlineTripletLoss_vis, self).__init__()
        self.margin = C.tri_loss_param['margin']
        self.triplet_selector = triplet_selector
        self.metric=C.tri_loss_param['metric']
        
    def forward(self, x, target):
        if type(x) in (tuple, list):
            x, embeddings = x
        else:
            embeddings = x
        
###        #**********************************
#        #For visualization
        # embeddings.requires_grad=True  # Creates computational graph (for each op we have a node with inputs and outputs)         
        x.register_hook(self.save_gradient)  # to extract the gradients from the last convolutional feature map to visualize the attention (intermediate variable)
###        #**********************************
        if self.metric == 'cosine':
            embeddings = embeddings / (1e-10 + torch.norm(embeddings, dim=1, keepdim=True))
            
        triplets = self.triplet_selector.get_triplets(embeddings, target, self.metric)
        
        if embeddings.is_cuda:
            triplets = triplets.cuda()

        if self.metric=='euclidean':
            ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1).sqrt()  # .pow(.5)
            an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1).sqrt()  # .pow(.5)
        else:
            anchors = embeddings[triplets[:, 0]]
            pos = embeddings[triplets[:, 1]]
            neg = embeddings[triplets[:, 2]]
            ap_distances = 1.0 - (anchors * pos).sum(1)
            an_distances = 1.0 - (anchors * neg).sum(1)
        losses = F.relu(ap_distances - an_distances + self.margin)
#        losses = ap_distances - an_distances + self.margin  # For visualization, discard hinge function(Relu)
        
#        compact_loss = (torch.abs(embeddings) - torch.mean(embeddings, 1, keepdim=True)).pow(2).sum(1)
#        compact_loss = compact_loss.mean()       
#        norm_loss = embeddings.pow(2).sum(1).sqrt().mean()
        
###        #*********************************************************************************
#        #For Visualization
        meanlosses=losses.mean()
        meanlosses.backward() # calculates gradient of mean losses w.r.t each embedding
#
###        #*********************************************************************************
        return losses.mean(), len(triplets), self.gradients # the latter is for calculating metrics   # Training

    def save_gradient(self, grad):
        self.gradients.append(grad)    
        
class OnlineTripletGain_vis(nn.Module):
    """
    Online Triplets Gain # Used for Visualizing the images via Grad-CAM
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, triplet_selector):
        self.gradients = []   # For Visualization
        super(OnlineTripletGain_vis, self).__init__()
        # self.margin = C.tri_loss_param['margin'] # No need of margin for triplet gain
        self.triplet_selector = triplet_selector
        self.metric=C.tri_loss_param['metric']
        
    def forward(self, x, target):
        if type(x) in (tuple, list):
            x, embeddings = x
        else:
            embeddings = x
        
###        #**********************************
#        #For visualization
        # embeddings.requires_grad=True  # Creates computational graph (for each op we have a node with inputs and outputs)         
        x.register_hook(self.save_gradient)  # to extract the gradients from the last convolutional feature map to visualize the attention (intermediate variable)
###        #**********************************
        if self.metric == 'cosine':
            embeddings = embeddings / (1e-10 + torch.norm(embeddings, dim=1, keepdim=True))
            
        triplets = self.triplet_selector.get_triplets(embeddings, target, self.metric)
        
        if embeddings.is_cuda:
            triplets = triplets.cuda()

        if self.metric=='euclidean':
            ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1).sqrt()  # .pow(.5)
            an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1).sqrt()  # .pow(.5)
        else:
            anchors = embeddings[triplets[:, 0]]
            pos = embeddings[triplets[:, 1]]
            neg = embeddings[triplets[:, 2]]
            ap_distances = 1.0 - (anchors * pos).sum(1)
            an_distances = 1.0 - (anchors * neg).sum(1)
        # Triplet Gain is G (a, p, n) = max (0, ||a, n|| - ||a,p||)
        gain= F.relu (an_distances - ap_distances)
        
###        #*********************************************************************************
#        #For Visualization
        meangain=gain.mean()
        meangain.backward() # calculates gradient of mean losses w.r.t each embedding
#
###        #*********************************************************************************
        return gain.mean(), len(triplets), self.gradients # the latter is for calculating metrics   # Training

    def save_gradient(self, grad):
        self.gradients.append(grad)    