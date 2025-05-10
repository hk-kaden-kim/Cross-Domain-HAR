import numpy as np
from numpy import copy

from scipy.special import softmax
from scipy.interpolate import interp1d

from sklearn.metrics import f1_score, cohen_kappa_score
from sklearn import decomposition, preprocessing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import umap
from tqdm import tqdm
import torch

RANDOM_SEED = 42


###################################################
# Open-Set related
###################################################
def create_oscr_curve(true_Y:np.array, pred_Y:np.array, logits:np.array):

    """ Calculates the OSCR values, iterating over the score of the target class of every sample,
    produces a pair (ccr, fpr) for every score.
    Args:
        gt (np.array): Integer array of target class labels.
        scores (np.array): Float array of dim [N_samples, N_classes] or [N_samples, N_classes+1]
        unk_label (int): Label to calculate the fpr, either negatives or unknowns. Defaults to -1 (negatives)
    Returns: Two lists first one for ccr, second for fpr.
    """
    # Change the unk_label to calculate for kn_unknown or unk_unknown
    gt = true_Y
    kn = (true_Y != 'unknown')
    unk = ~kn

    # Get total number of samples of each type
    total_kn = np.sum(kn)
    total_unk = np.sum(unk)

    ccr, fpr, thr = [], [], []
    scores = softmax(logits, axis=1)
    pred_class = pred_Y
    max_score = np.max(scores, axis=1)
    # target_score = scores[kn][range(kn.sum()), gt[kn]]
    target_score = max_score[kn]
    # print(target_score, len(target_score))
    for tau in tqdm(np.unique(target_score)[:-1], mininterval=30):
        thr.append(tau)

        val = ((pred_class[kn] == gt[kn]) & (target_score > tau)).sum() / total_kn
        ccr.append(val)

        val = (unk & (max_score > tau)).sum() / total_unk
        fpr.append(val)


    ccr = np.array(ccr)
    fpr = np.array(fpr)
    thr = np.array(thr)
    return ccr, fpr, thr

def create_balanced_oscr_curve(true_Y:np.array, pred_Y:np.array, logits:np.array):

    """ Calculates the OSCR values, iterating over the score of the target class of every sample,
    ...
    """
    # Change the unk_label to calculate for kn_unknown or unk_unknown
    gt = true_Y
    kn = (true_Y != 'unknown')
    unk = ~kn


    # Get total number of samples of each type
    total_unk = np.sum(unk)
    kn_label_set, total_kn_label_set = np.unique(true_Y[kn], return_counts=True)

    ccr, fpr, thr = [], [], []
    scores = softmax(logits, axis=1)
    pred_class = pred_Y
    max_score = np.max(scores, axis=1)
    target_score = max_score[kn]

    for tau in tqdm(np.unique(target_score)[:-1], mininterval=30):

        thr.append(tau)

        # (Macro) balanced ccr by label
        ccr_by_label = []
        # calculate ccr by each label
        for idx, kn_label in enumerate(kn_label_set):
            mask = (gt==kn_label)
            # get predictions and max_scores where the ground truth is kn_label
            pred_masked = pred_class[mask]
            max_score_masked = max_score[mask]
            # count 'a prediction is same with kn_label' and 'max_score is greater than threshold'
            # divide by the count of the ground truth which is kn_label
            val = ((pred_masked == kn_label) & (max_score_masked > tau)).sum() 
            val = val / total_kn_label_set[idx]
            ccr_by_label.append(val)
        # final ccr is the average of each label's ccr
        ccr.append(np.average(np.array(ccr_by_label)))

        # fpr
        val = (unk & (max_score > tau)).sum() / total_unk
        fpr.append(val)


    ccr = np.array(ccr)
    fpr = np.array(fpr)
    thr = np.array(thr)
    return ccr, fpr, thr

def create_balanced_oscr_curve_gpu(true_Y:np.array, pred_Y:np.array, logits:np.array, my_device):

    """ Calculates the OSCR values, iterating over the score of the target class of every sample,
    ...
    """
    # Change label type
    transdict={
                'sit-stand': 0,
                'sleep': 1,
                'mixed': 2,
                'walking': 3,
                'vehicle': 4,
                'bicycling': 5,
                'unknown': -1,
            }
    true_Y = np.array([transdict[l] for l in true_Y])
    pred_Y = np.array([transdict[l] for l in pred_Y])
    
    # Change numpy to torch
    true_Y = torch.from_numpy(true_Y).to(my_device, dtype=torch.int)
    pred_Y = torch.from_numpy(pred_Y).to(my_device, dtype=torch.int)

    # Change the unk_label to calculate for kn_unknown or unk_unknown
    gt = true_Y
    kn = (true_Y != transdict['unknown'])
    unk = ~kn

    # Get total number of samples of each type
    total_unk = sum(unk)
    kn_label_set, total_kn_label_set = torch.unique(true_Y[kn], return_counts=True)

    ccr, fpr, thr = [], [], []
    scores = softmax(logits, axis=1)
    pred_class = pred_Y
    max_score = np.max(scores, axis=1)
    target_score = max_score[kn.tolist()]

    max_score = torch.from_numpy(max_score).to(my_device, dtype=torch.float)
    for tau in tqdm(np.unique(target_score)[:-1], mininterval=30):

        # balanced ccr by label
        ccr_by_label = []
        # calculate ccr by each label
        for idx, kn_label in enumerate(kn_label_set):
            mask = (gt==kn_label)
            # get predictions and max_scores where the ground truth is kn_label
            pred_masked = pred_class[mask]
            max_score_masked = max_score[mask]
            # count 'a prediction is same with kn_label' and 'max_score is greater than threshold'
            # divide by the count of the ground truth which is kn_label
            val = ((pred_masked == kn_label) & (max_score_masked > tau)).sum() 
            val = val / total_kn_label_set[idx]
            val = val.item()
            ccr_by_label.append(val)
        # final ccr is the average of each label's ccr
        ccr.append(np.average(np.array(ccr_by_label)))

        # fpr
        val = (unk & (max_score > tau)).sum() / total_unk
        val = val.item()
        fpr.append(val)
        tau = tau.item()
        thr.append(tau)

        # print(ccr_by_label, val, tau)
        # print(type(ccr_by_label), type(val), type(tau))
        # break

    ccr = np.array(ccr)
    fpr = np.array(fpr)
    thr = np.array(thr)
    return ccr, fpr, thr

def find_ccr_at_fpr(FPR:np.array, CCR:np.array, ref_fpr:float):
    f = interp1d( FPR, CCR )
    ccr = f(ref_fpr).item()
    return ccr

def find_thr_at_fpr(FPR:np.array, Threshold:np.array, ref_fpr:float):
    f = interp1d( FPR, Threshold )
    thr = f(ref_fpr).item()
    return thr

def cal_softmax_score(logits:np.array):
    return np.max(softmax(logits, axis=1), axis=1)

def cua_oscr_range(cua_1:str, cua_2:str, is_target=False):
    prefix = ''
    if is_target: prefix = 't_'
    CCR_1 = np.load(f'./_inference/oscr_result/gpu/{prefix}{cua_1}_CCR.npy')
    FPR_1 = np.load(f'./_inference/oscr_result/gpu/{prefix}{cua_1}_FPR.npy')

    CCR_2 = np.load(f'./_inference/oscr_result/gpu/{prefix}{cua_2}_CCR.npy')
    FPR_2 = np.load(f'./_inference/oscr_result/gpu/{prefix}{cua_2}_FPR.npy')

    xfill_min = np.max([np.min(FPR_1), np.min(FPR_2)])
    xfill_max = np.min([np.max(FPR_1), np.max(FPR_2)])
    xfill = np.concatenate([FPR_1, FPR_2])
    xfill = np.unique(xfill)
    xfill = np.sort(xfill)
    xfill = xfill[(xfill>=0.01)&(xfill<=xfill_max)]

    y1fill = []
    y2fill = []
    for x in xfill:
        y = find_ccr_at_fpr(FPR_1, CCR_1, x)
        y1fill += [y]
        y = find_ccr_at_fpr(FPR_2, CCR_2, x)
        y2fill += [y]

    fpr_fill = xfill
    ccr1_fill = y1fill
    ccr2_fill = y2fill
    
    return fpr_fill, ccr1_fill, ccr2_fill


###################################################
# Closed-Set related
###################################################
def cal_bal_accuracy(true_Y:np.array, pred_Y:np.array):

    # kn = ( true_Y != 'unknown' )
    # true_Y = true_Y[kn]
    kn_label_set, total_kn_label_set = np.unique(true_Y, return_counts=True)

    # balanced ccr by label
    ccr_by_label = []
    # calculate ccr by each label
    for idx, kn_label in enumerate(kn_label_set):
        mask = (true_Y==kn_label)
        # get predictions and max_scores where the ground truth is kn_label
        pred_masked = pred_Y[mask]
        # count 'a prediction is same with kn_label' and 'max_score is greater than threshold'
        # divide by the count of the ground truth which is kn_label
        val = (pred_masked == kn_label).sum() 
        val = val / total_kn_label_set[idx]
        ccr_by_label.append(val)
    # final ccr is the average of each label's ccr
    accuracy = np.average(np.array(ccr_by_label))

    return accuracy

def multi_class_eval(true_Y:np.array, pred_Y:np.array):
    f1 = f1_score(true_Y, pred_Y, average='macro')
    k = cohen_kappa_score(true_Y, pred_Y)
    acc = cal_bal_accuracy(true_Y, pred_Y) # Accuracy
    return f1, k, acc

def get_confusionMatrix(true_Y:np.array, pred_Y:np.array):

    labels = ['sit-stand','sleep','mixed','walking','bicycling','vehicle']
    simple_labels = ['ss','sl','mx','wk','bc','vh']
    mappings = dict(zip(labels, simple_labels))

    true_Y_simple = copy(true_Y)
    for k, v in mappings.items(): true_Y_simple[true_Y==k] = v

    pred_Y_simple = copy(pred_Y)
    for k, v in mappings.items(): pred_Y_simple[pred_Y==k] = v

    cm = confusion_matrix(true_Y_simple, pred_Y_simple, labels=simple_labels, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                            display_labels=simple_labels,)
    return disp


###################################################
# Semantic Analysis
###################################################
def sampling_by_label(X,Y,ratio=None,samples=None,label_balance=True,is_verbose=False):

    np.random.seed(RANDOM_SEED)

    total_num = len(Y)
    ulabel = np.unique(Y)
    idxs = []
    
    if samples:
        if is_verbose: print(f"Random sampling for each label to make total {samples}ea")
        
        for i, l in enumerate(ulabel):
            l_idx = np.where(Y == l)[0]
            l_sample_num = int(samples * ( len(l_idx) / total_num ))
            if label_balance:
                l_sample_num = samples//len(ulabel)
            l_idx_filter = np.random.choice(l_idx, l_sample_num).tolist()
            idxs += l_idx_filter
            if is_verbose:
                print(f"{l}: {len(l_idx_filter)} ea ({100*len(l_idx_filter)/len(l_idx):.2f} %)")
                

    if ratio:
        if is_verbose: print(f"Random sampling {ratio*100}% for each label")
        
        for i, l in enumerate(ulabel):
            l_idx = np.where(Y == l)[0]
            l_sample_num = int(len(l_idx) * ratio )
            l_idx_filter = np.random.choice(l_idx, l_sample_num).tolist()
            idxs += l_idx_filter
            if is_verbose:
                print(f"{l}: {len(l_idx_filter)} ea")
                
    if is_verbose: print(f"Final samples: {len(idxs)} ea\nDone")

    return X[idxs], Y[idxs]

def reduceme(X, PCA=True, UMAP=True):
    ########################################
    # Reduce dimensionality pipeline
    # PCA - UMAP
    ########################################
    """
    UMAP
    Uniform Manifold Approximation and Projection (UMAP) is a dimension reduction technique
    that can be used for visualisation similarly to t-SNE, but also for general non-linear dimension reduction.
    https://umap-learn.readthedocs.io/en/latest/basic_usage.html
    https://umap-learn.readthedocs.io/en/latest/faq.html#what-is-the-difference-between-pca-umap-vaes
    https://github.com/lmcinnes/umap
    https://arxiv.org/abs/1802.03426
    """
    if PCA:
        pca = decomposition.PCA(n_components=50)
        X_pca = pca.fit_transform(X)
        # print("\t\t\tPCA : ", X_pca.shape)
    else:
        X_pca = X

    if UMAP:
        scaler = preprocessing.StandardScaler()
        X_scaled = scaler.fit_transform(X_pca.reshape(X.shape[0], -1))
        reducer = umap.UMAP(random_state=RANDOM_SEED, n_jobs=1)
        X_red = reducer.fit_transform(X_scaled)
    else:
        X_red = X_pca

    return X_red

def reduceme_3D(X, PCA=True, UMAP=True):
    ########################################
    # Reduce dimensionality pipeline
    # PCA - UMAP
    ########################################
    """
    UMAP
    Uniform Manifold Approximation and Projection (UMAP) is a dimension reduction technique
    that can be used for visualisation similarly to t-SNE, but also for general non-linear dimension reduction.
    https://umap-learn.readthedocs.io/en/latest/basic_usage.html
    https://umap-learn.readthedocs.io/en/latest/faq.html#what-is-the-difference-between-pca-umap-vaes
    https://github.com/lmcinnes/umap
    https://arxiv.org/abs/1802.03426
    """
    if PCA:
        pca = decomposition.PCA(n_components=50)
        X_pca = pca.fit_transform(X)
        # print("\t\t\tPCA : ", X_pca.shape)
    else:
        X_pca = X

    if UMAP:
        scaler = preprocessing.StandardScaler()
        X_scaled = scaler.fit_transform(X_pca.reshape(X.shape[0], -1))
        reducer = umap.UMAP(random_state=RANDOM_SEED, n_jobs=1, n_components=3)
        X_red = reducer.fit_transform(X_scaled)
    else:
        X_red = X_pca

    return X_red
