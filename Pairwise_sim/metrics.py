# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 15:11:53 2020

@author: Sivapriyaa
"""
import numpy as np
import torch
#from config import Configuration as C

C = None

def set_config(_C):
    global C
    C = _C

class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target, loss):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError
        
        
class MeanLoss(Metric):
    def __init__(self):
        self.reset()

    def __call__(self, *args):
        if len(args) == 1:
            self.values = np.append(self.values, args[0])
        else:
            x, target, loss = args
            self.values = np.append(self.values, loss.item())

    def value(self):
        if len(self.values)==0:
            return float('inf')
        else:
            return np.mean(self.values)

    def reset(self):
        self.values = np.array([])

    def name(self):
        return 'MeanLoss'

#class EmdRecorder(Metric):
#    def __init__(self, num_emd):
#        self.set_len = num_emd
#        self.insert_idx = [0, C.test_batch_size]
#
#        self.sketch_emd = np.empty(shape=[num_emd, C.emd_len])
#        self.photo_emd = np.empty(shape=[num_emd, C.emd_len])
#
#        self.sketch_id = np.empty(shape=[num_emd])
#        self.photo_id = np.empty(shape=[num_emd])
#
#        self.sketch_img = np.empty(shape=[num_emd, C.img_channel, C.img_size, C.img_size])
#        self.photo_img = np.empty(shape=[num_emd, C.img_channel, C.img_size, C.img_size])
#
#    def __call__(self, sketch_emd, photo_emd, sketch_id, photo_id, sketch_img, photo_img):
#        # Append data
#        self.sketch_emd[self.insert_idx[0]:self.insert_idx[1]] = sketch_emd
#        self.photo_emd[self.insert_idx[0]:self.insert_idx[1]] = photo_emd
#
#        self.sketch_id[self.insert_idx[0]:self.insert_idx[1]] = sketch_id
#        self.photo_id[self.insert_idx[0]:self.insert_idx[1]] = photo_id
#
#        self.sketch_img[self.insert_idx[0]:self.insert_idx[1]] = sketch_img
#        self.photo_img[self.insert_idx[0]:self.insert_idx[1]] = photo_img
#
#        # Update index
#        self.insert_idx += [C.batch_size, C.batch_size]
#
#    def value(self):
#        embedding = np.concatenate((self.sketch_emd, self.photo_emd), axis=0)
#        metadata = np.concatenate((self.sketch_id, self.photo_id), axis=0)
#        label_img = np.concatenate((self.sketch_img, self.photo_img), axis=0)
#
#        self._get_emd_distance()
#        return embedding, metadata, label_img
#
#    def _get_emd_distance(self, sketch_emd, photo_emd):
#        """
#        Calculate euclidean distances between sketch and photo embedding vectors.
#        Args:
#            sketch_emd (torch.Tensor): [Num_embeddings, embedding_vector_length]
#            photo_emd  (torch.Tensor): [Num_embeddings, embedding_vector_length]
#        Returns:
#            distance_matrix (torch.Tensor): [Num_embeddings, Num_embeddings]
#        """
#        distance_matrix = -2 * vectors.mm(torch.t(vectors)) + \
#                          vectors.pow(2).sum(dim=1).view(1, -1) + \
#                          vectors.pow(2).sum(dim=1).view(-1, 1)
#        distance_matrix = distance_matrix.sqrt()
#
#        a_ = np.re(np.power(a, 2), axis=1)
#
#        return distance_matrix
#
#    def reset(self):
#        self.insert_idx = [0, C.batch_size]
#
#        self.sketch_emd = np.empty(shape=[self.set_len, C.emd_len])
#        self.photo_emd = np.empty(shape=[self.set_len, C.emd_len])
#
#        self.sketch_id = np.empty(shape=[self.set_len])
#        self.photo_id = np.empty(shape=[self.set_len])
#
#        self.sketch_img = np.empty(shape=[self.set_len, C.img_channel, C.img_size, C.img_size])
#        self.photo_img = np.empty(shape=[self.set_len, C.img_channel, C.img_size, C.img_size])
#
#    def name(self):
#        return 'Embeddings records'


class AccumulatedAccuracyMetric(Metric):
    """
    Works with classification model
    """

    def __init__(self):
        self.correct = 0
        self.total = 0

    def __call__(self, outputs, target, loss):
        pred = outputs[0].data.max(1, keepdim=True)[1]
        self.correct += pred.eq(target[0].data.view_as(pred)).cpu().sum()
        self.total += target[0].size(0)
        return self.value()

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return 100 * float(self.correct) / self.total

    def name(self):
        return 'Accuracy'


class AverageNonzeroTripletsMetric(Metric):
    '''
    Counts average number of nonzero triplets found in minibatches
    '''

    def __init__(self):
        self.values = []

    def __call__(self, outputs, target, loss):
        self.values.append(loss[1])
        return self.value()

    def reset(self):
        self.values = []

    def value(self):
        return np.mean(self.values)

    def name(self):
        return 'Average nonzero triplets'

class OSPAccuracyMetric(Metric):
    """
    Gives accuracies measure based on the output of the networks incorporating 
    the Othogonal Space Projection scheme for cross-modal face matching. It 
    calculates classification accuracies for respective face modalities (
    currently two modalities are supported). It also returns accuracy for cross
    -modal face verification. 
    """
    
    def __init__(self):
        self.reset()
        
    def __call__(self, outputs, target, loss):
        x, emd = outputs
        x_c, y_c, x_p, x_q, W, P, Q = x
        y, types = target[:,0], target[:,1]
        x_p = x_p[types==0] # composite
        y_p = y[types==0]
        x_q = x_q[types==1] # photo
        y_q = y[types==1]
        
        X = [x_c, x_p, x_q]
        Y = [y_c, y_p, y_q]
        
        for i in range(3):
            x = torch.nn.functional.softmax(X[i], dim=1)
            pred = x.data.max(1, keepdim=True)[1]
            self.correct[i] += pred.eq(Y[i].data.view_as(pred)).cpu().sum()
            self.total[i] += Y[i].size(0)
        
        self.orthoLosses.append(loss[1].item())
        return self.value()
        
    def reset(self):
        self.correct = np.zeros((3,), dtype=np.int)
        self.total = np.zeros((3,), dtype=np.int)
        self.orthoLosses = []
        
    def value(self):
        acc = 100 * self.correct.astype(dtype=np.float) / self.total
        return tuple([*acc, np.mean(self.orthoLosses)]) 
    
    def name(self):
        return ['Cross-modal verif acc.', 'Composite ident. acc.',
                'Photo ident. acc.', 'Mean ortho. losses']
        
class CDLAccuracyMetric(Metric):
    """
    Gives accuracies measure based on the output of the networks incorporating 
    cross entropy, triplet and trace norm losses for cross-modal face matching.  
    """
    
    def __init__(self, stratified_split=True):
        self.stratified_split = stratified_split
        self.reset()
        
    def __call__(self, outputs, target, loss):
        x, emd = outputs
        x, x_n, x_v, W_n, W_v = x
        y, types = target[:,0].long(), target[:,1]
        x_n = x_n[types==0] # composite
        y_n = y[types==0]
        x_v = x_v[types==1] # photo
        y_v = y[types==1]
        
        X = [x_n, x_v]
        Y = [y_n, y_v]
        
        for i in range(2):
            x = torch.nn.functional.softmax(X[i], dim=1)
            pred = x.data.max(1, keepdim=True)[1]
            self.correct[i] += pred.eq(Y[i].data.view_as(pred)).cpu().sum()
            self.total[i] += Y[i].size(0)
        
        self.n_triplet.append(loss[-1])
        self.losses.append([loss[i].cpu().detach().numpy().tolist() for i in range(len(loss)-1)])
        
        return self.value()
        
    def reset(self):
        self.correct = np.zeros((2,), dtype=np.int)
        self.total = np.zeros((2,), dtype=np.int)
        self.losses = []
        self.n_triplet = []
        
    def value(self):
        if self.stratified_split:     # No overlap between Training classes and Validation classes
            acc = 100 * self.correct.astype(dtype=np.float) / self.total
            return tuple([*acc, *np.mean(self.losses, axis=0).tolist(), np.mean(self.n_triplet)]) 
        else:
            return tuple([*np.mean(self.losses, axis=0).tolist(), np.mean(self.n_triplet)])
    
    def name(self):
        if self.stratified_split:
            return ['CompositeIdentAcc.',
                'PhotoIdentAcc.', 'TotalLoss', 'CELoss', 'TNLoss', 'OrthoLoss', 'TripletLoss', 'NumberOfTriplets'] # These values correspond to the value() function above
        else:
            return ['TotalLoss', 'CELoss', 'TNLoss', 'OrthoLoss', 'TripletLoss', 'NumberOfTriplets']
        

class Metric_ce_tri(Metric):
    """
    Metrics 
    """
    
    def __init__(self):
        self.reset()
        
    def __call__(self, outputs, target, loss):
        x, emd = outputs
        y = target[:,0].long()
        x = torch.nn.functional.softmax(x, dim=1)
        pred = x.data.max(1, keepdim=True)[1]
        self.correct += pred.eq(y.data.view_as(pred)).cpu().sum()
        self.total += y.size(0)
        self.n_triplet.append(loss[-1])
        self.losses.append([loss[i].item() for i in range(len(loss)-1)])
        
        return self.value()
        
    def reset(self):
        self.correct = 0
        self.total = 0
        self.losses = []
        self.n_triplet = []
        
    def value(self):
        acc = 100 * float(self.correct) / self.total
        return tuple([acc, *np.mean(self.losses, axis=0).tolist(), np.mean(self.n_triplet)]) 
    
    def name(self):
        return ['IdentAcc.', 'TotalLoss', 'CELoss', 'TripletLoss', 'NumberOfTriplets']


class Metric_agt(Metric):
    """
      
    """
    
    def __init__(self):
        self.reset()
        
    def __call__(self, outputs, target, loss):
        self.n_triplet.append(loss[-1])
        self.losses.append([loss[i].cpu().detach().numpy().tolist() for i in range(len(loss)-1)])       
        return self.value()
        
    def reset(self):
        self.losses = []
        self.n_triplet = []
        
    def value(self):
        return tuple([*np.mean(self.losses, axis=0).tolist(), np.mean(self.n_triplet)]) 
    
    def name(self):
        return ['TotalLoss', 'TripletLoss', 'NormLoss', 'NumberOfTriplets']   


class Metric_osp_wass_tn(Metric):
    """
    For recording performance metrics of models trained with OSP_WASS_TN_Loss
    """
    
    def __init__(self):
        self.reset()
        
    def __call__(self, outputs, target, loss):
        x, emd = outputs
        x_c, x_p, x_q, W, P, Q, F_n, F_v = x
        y, types = target[:,0], target[:,1]
        x_p = x_p[types==0] # composite
        y_p = y[types==0]
        x_q = x_q[types==1] # photo
        y_q = y[types==1]
        
        X = [x_p, x_q]
        Y = [y_p, y_q]
        
        for i in range(2):
            x = torch.nn.functional.softmax(X[i], dim=1)
            pred = x.data.max(1, keepdim=True)[1]
            self.correct[i] += pred.eq(Y[i].data.view_as(pred)).cpu().sum()
            self.total[i] += Y[i].size(0)
        
        self.losses.append([x.item() for x in loss[1:]])
        return self.value()
        
    def reset(self):
        self.correct = np.zeros((2,), dtype=np.int)
        self.total = np.zeros((2,), dtype=np.int)
        self.losses = []
        
    def value(self):
        acc = 100 * self.correct.astype(dtype=np.float) / self.total
        return tuple([*acc, *np.mean(self.losses, axis=0).tolist()]) 
    
    def name(self):
        return ['MeanCompositeIdentAcc.',
                'MeanPhotoIdentAcc.', 'MeanCELoss', 'MeanWassLoss', 'MeanTNLoss', 'MeanOrthoMatLoss', 'MeanOSPLoss']     

class EmbedDistance(object):
    """
    Borrowed from Fangliang
    """
    def __init__(self):
        pass

    def __call__(self):
        pass

    def value(self, sketch_emd, photo_emd):
        return self._get_emd_distance(sketch_emd, photo_emd)

    def _get_emd_distance(self, sketch_emd, photo_emd):
        """
        Calculate euclidean distances between sketch and photo embedding vectors.
        Args:
            sketch_emd (numpy.array): [Num_sketch, embedding_length]
            photo_emd  (numpy.array): [Num_photo, embedding_length]
        Returns:
            distance_matrix (numpy.array): [Num_sketch, Num_photo]
        """

        # <editor-fold desc="[Option 1] Centralise sketch and photo embeddings separately ...">
        centroid_vector = np.mean(sketch_emd, axis=0, keepdims=True)
        if len(sketch_emd) > 1:
            sketch_emd = sketch_emd - centroid_vector
        sketch_emd = sketch_emd / np.linalg.norm(sketch_emd, axis=-1, keepdims=True)

        centroid_vector = np.mean(photo_emd, axis=0, keepdims=True)
        if len(photo_emd) > 1:
            photo_emd = photo_emd - centroid_vector
        photo_emd = photo_emd / np.linalg.norm(photo_emd, axis=-1, keepdims=True)
        # </editor-fold>

        # <editor-fold desc="[Option 2] Centralise sketch and photo embeddings together ...">
        # num_sketch = len(sketch_emd)
        # num_photo = len(photo_emd)
        # all_emd = np.concatenate((sketch_emd,photo_emd), axis=0)
        # centroid_vector = np.mean(all_emd, axis=0, keepdims=True)
        # all_emd = all_emd - centroid_vector
        # all_emd = all_emd / np.linalg.norm(all_emd, axis=-1, keepdims=True)
        # sketch_emd, photo_emd, _ = np.split(all_emd, [num_sketch, num_sketch+num_photo])
        # </editor-fold>

        # <editor-fold desc="[+] Calculate euclidean distance ...">
        a2 = np.reshape(np.sum(np.power(sketch_emd, 2), axis=1), [-1, 1])
        b2 = np.reshape(np.sum(np.power(photo_emd, 2), axis=1), [1, -1])
        ab = -2 * np.matmul(sketch_emd, np.transpose(photo_emd))
        distance_matrix = ab + a2 + b2
        distance_matrix = np.sqrt(distance_matrix)
        # </editor-fold>

        return distance_matrix

    def reset(self):
        pass

    def name(self):
        pass


class CumulMatchCurve(object):
    """
    Borrowed from Fangliang
    """    
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass

    @staticmethod
    def value(dist_mtrx, sketch_id, photo_id):
        """
        Calculate the cumulative match curve.
        Args:
            dist_mtrx (numpy.array): [Num_sketch, Num_photo]
            sketch_id (numpy.array): [Num_sketch]
            photo_id (numpy.array) : [Num_photo]

        Returns:
            cumul_match_curve_x (numpy.array): [cumul_match_curve]
            cumul_match_curve_y (numpy.array): [cumul_match_curve]
        """
        distance_sort_idx = np.argsort(dist_mtrx, axis=-1)  # [Num_sketch, Num_photo]
        rank_matched = np.zeros(sketch_id.shape[0])  # [Num_sketch]

        for i, s_id in enumerate(sketch_id):
            label = np.argwhere(s_id == photo_id).reshape([-1])
            rank = np.where(distance_sort_idx[i, :] == label)
            rank_matched[i] = rank[0]

        unique_rank, rank_cnt, = np.unique(rank_matched, return_counts=True)
        num_each_rank = np.zeros([len(photo_id)])  # [Num_photo]
        num_each_rank[unique_rank.astype(int)] = rank_cnt
        cumulated_num_rank = np.zeros([len(photo_id)])  # [Num_photo]
        for i in range(len(num_each_rank)):
            cumulated_num_rank[i] += np.sum(num_each_rank[:i + 1])

        # Calculate curve (x, y) for plotting
        cumul_match_curve_y = cumulated_num_rank / len(sketch_id)
        cumul_match_curve_x = np.arange(1, 1 + cumul_match_curve_y.shape[0])
        return cumul_match_curve_x, cumul_match_curve_y

    def reset(self):
        pass

    def name(self):
        pass
    
class OnlineClassificationAccMetric(Metric):
    """
    Works with classification model
    """

    def __init__(self):
        self.correct = 0
        self.total = 0

    def __call__(self, outputs, target, loss):
        pred = outputs[0].max(1, keepdim=True)[1]
        y = outputs[1]
        self.correct += pred.eq(y.view_as(pred)).cpu().sum()
        self.total += y.size(0)
        return self.value()

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return 100 * float(self.correct) / self.total

    def name(self):
        return 'Accuracy'
    
class MseTriLossMetric(Metric):
    """
    For recording performance metrics of models trained with MSE_TRI_Loss
    """
    
    def __init__(self):
        self.reset()
        
    def __call__(self, outputs, target, loss):
        self.n_triplet.append(loss[-1])
        self.losses.append([loss[i].cpu().detach().numpy().tolist() for i in range(len(loss)-1)])
        return self.value()
        
    def reset(self):
        self.losses = []
        self.n_triplet = []
        
    def value(self):
        return tuple([*np.mean(self.losses, axis=0).tolist(), np.mean(self.n_triplet)]) 
    
    def name(self):
        return ['TotalLoss', 'MseLoss', 'TripletLoss', 'NumberOfTriplets']
   
    
class WeightedTriLossMetric(Metric):
    """
    For recording performance metrics of models trained with MSE_TRI_Loss
    """
    
    def __init__(self):
        self.reset()
        
    def __call__(self, outputs, target, loss):
        self.n_triplet.append(loss[-1])
        self.losses.append([loss[i].cpu().detach().numpy().tolist() for i in range(len(loss)-1)])
        return self.value()
        
    def reset(self):
        self.losses = []
        self.n_triplet = []
        
    def value(self):
        return tuple([*np.mean(self.losses, axis=0).tolist(), np.mean(self.n_triplet)]) 
    
    def name(self):
        return ['TotalLoss', 'TriLoss', 'RatingMSELoss', 'NumberOfTriplets']