from sklearn.metrics import auc, roc_curve
import numpy as np
import torch

def auc_acc(pred, true):
    sigma = 1e-6
    true = true.astype(int)
    pred = pred + np.random.rand(*pred.shape) * sigma
    auc_list = []
    if pred.shape[0] != true.shape[0]:
        for iter1 in range(pred.shape[0]):
            id = pred[iter1].argsort()
            for iter2 in range(true.shape[0]):
                fpr, tpr, thresholds = roc_curve(true[iter2][id], pred[iter1][id], pos_label=1)
                auc_list.append(auc(fpr, tpr))
    else:
        for iter1 in range(pred.shape[0]):
            id = pred[iter1].argsort()
            fpr, tpr, thresholds = roc_curve(true[iter1][id], pred[iter1][id], pos_label=1)
            auc_list.append(auc(fpr, tpr))
    return np.mean(np.array(auc_list))


def KL_Divengence(P, Q, multi_state=False, log=torch.log2):
    if multi_state:
        P /= P.sum(-1, keepdims=True)
        Q /= Q.sum(-1, keepdims=True)
        out = (P * (log(P) - log(Q))).sum(-1)
    else:
        out = P * (log(P) - log(Q)) + (1 - P) * (log(1 - P) - log(1 - Q))
    return out

def JS_Divengence(P, Q, multi_state=False):
    M = (P + Q) / 2
    return (KL_Divengence(P, M, multi_state=False) + KL_Divengence(Q, M, multi_state=False)) / 2


def CrossEntropy(pred, true, multi_state=True, sigma=1e-8):
    assert((pred.max() <= 1) and (pred.min() >= 0))
    assert((true.max() <= 1) and (true.min() >= 0))
    true = true.clamp(sigma, 1-sigma)
    pred = pred.clamp(sigma, 1-sigma)
    if not multi_state:
        return (- true * torch.log(pred) - (1 - true) * torch.log(1 - pred))
    else:
        return (- true * torch.log(pred)).sum(-1)


def EffectiveInformation(
    pred, true, multi_state=False, sigma=1e-10, dtype=None): # input: [number of samples, number of vertices]
    
    assert((pred.max() <= 1) and (pred.min() >= 0))
    assert((true.max() <= 1) and (true.min() >= 0))
    pred = pred.float().clamp(sigma, 1-sigma)
    true = true.float().clamp(sigma, 1-sigma)
    
    if multi_state:
        mean = true.clone()
        for iter in range(len(true.shape)-1):
            mean = mean.mean(iter, keepdims=True)
        temp1, temp2 = KL_Divengence(mean, true, multi_state=True).mean(), KL_Divengence(pred, true, multi_state=True).mean()
    else:
        mean = true * 0.0 + true.mean()
        #pred /= pred.sum(-1, keepdim=True)
        #true /= true.sum(-1, keepdim=True)
        #mean /= mean.sum(-1, keepdim=True)
        temp1, temp2 = KL_Divengence(mean, true, multi_state=False).mean(), KL_Divengence(pred, true, multi_state=False).mean()
        #print(temp1)
        #print(temp2)
    if dtype:
        temp1, temp2 = dtype(temp1), dtype(temp2)
    return temp1, temp2, temp1 - temp2 # [Mean:True, Pred:True, Mean:True-Pred:True]

def CorrelationCoefficient(pred, true, dtype=None, sigma=1e-12):
    x = pred.float()
    y = true.float()

    vx = x - x.mean(dim=-1, keepdims=True)
    vy = y - y.mean(dim=-1, keepdims=True)

    corr = torch.sum(vx * vy, dim=-1, keepdims=True) / ((torch.sqrt(torch.sum(vx ** 2, dim=-1, keepdims=True)) * torch.sqrt(torch.sum(vy ** 2, dim=-1, keepdims=True))) + sigma)
    return corr