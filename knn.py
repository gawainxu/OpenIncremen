#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 10:11:36 2022

@author: zhi
"""

import torch
import pickle
import numpy as np
from collections import Counter
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from data_loader import iCIFAR100, iCIFAR10, mnist
from resnet_big import SupConResNet
from mlp import MLP
from itertools import chain

from metric import accuracy, compareLabelsOSR


def normalFeatureReading(data_loader, model):
    features = []
    labels = []

    for i, (img, l) in enumerate(data_loader):

        output = model(img)
        features.append(output.detach().numpy())
        labels.append(l.item())
        
    return features, labels
        

def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]


def KNN(test_feature, exemplars, K):
    
    exemplar_features, exemplar_labels = exemplars
    
    # calculate similarity
    # test_feature = np.tile(test_feature, (len(exemplar_features), 1))
    # similarities = np.squeeze(np.matmul(np.array(exemplar_features), test_feature.T))[:,0]
    similarities = []
    for ef in exemplar_features:
        similarity = np.matmul(np.array(ef), test_feature.T)
        similarities.append(similarity)
    similarities = np.squeeze(np.array(similarities))
    
    ind = np.argsort(similarities)[-K:]
    closest_labels = []
    for i in ind:
        closest_labels.append(exemplar_labels[i].item())
    
    closest_class = most_frequent(closest_labels)
    
    return closest_class


def OSNN(test_feature, exemplars, K):
    
    exemplar_features, exemplar_labels = exemplars
    
    similarities = []
    for ef in exemplar_features:
        similarity = np.matmul(np.array(ef), test_feature.T)
        similarities.append(similarity)
    similarities = np.squeeze(np.array(similarities))
    
    ind = np.argsort(similarities)[-K:]
    closest_labels = []
    for i in ind:
        closest_labels.append(exemplar_labels[i].item())
    
    occurence_count = Counter(closest_labels)
    if len(occurence_count) == 1:
        if sum(similarities[ind])/len(ind) > Ts:
            return 10, occurence_count.most_common(2)[0][0], sum(similarities[ind])/len(ind)                         # TODO the return number
        else:
            return 10, 1000, sum(similarities[ind])/len(ind)
    closest_class, second_closest_class = occurence_count.most_common(2)[0][0], occurence_count.most_common(2)[1][0]
    i1 = [i for i, x in enumerate(closest_labels) if closest_labels[i] == closest_class]
    i2 = [i for i, x in enumerate(closest_labels) if closest_labels[i] == second_closest_class]
    closest_ind = ind[i1]
    second_closest_ind = ind[i2]
    closest_similarity = np.sum(similarities[closest_ind])
    second_closest_similarity = np.sum(similarities[second_closest_ind])
    
    R = closest_similarity / second_closest_similarity
    if R < Tr:                                                
        return R, 1000, sum(similarities[ind])/len(ind)
    else:
        return R, closest_class, sum(similarities[ind])/len(ind)
    

def compareLabels(estLabels, trueLabels):
    
    assert len(estLabels) == len(trueLabels)
    unEquals = 0
    for i in range(len(estLabels)):
        if estLabels[i] != trueLabels[i]:
            unEquals += 1
            
    return unEquals



if __name__ == "__main__":
    
    Ts = 0.85
    Tr = 1.9
    
    K = 10
    num_classes = 20
    classes = [i for i in range(num_classes)]  +  [i for i in range(90, 100)]                            ####
    dataset = "cifar100"
    exemplar_feature_path  = "./features/exemplar_cifar100_class_20_resnet18_memorysize_50_alfa_0.2_temp_0.05_mem_2000"   
    
    model = SupConResNet("resnet18")
    ckpt = torch.load("./save/SupCon_cifar100_class_20_resnet18_lr_0.001_epoch_600_bsz_512_temp_0.05_alfa_0.2_mem_2000_incremental/last.pth", map_location='cpu')
    state_dict = ckpt['model']

    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v

    state_dict = new_state_dict
    model = model.cpu()
    model.load_state_dict(state_dict)
    model.eval()
    
    with open(exemplar_feature_path, "rb") as f:
        exemplar_features, exemplar_labels = pickle.load(f)
        

    if dataset == "cifar100":
        transform = transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                             (0.2675, 0.2565, 0.2761)),])
        test_set = iCIFAR100(root='../datasets', train=False,                        
                             classes=classes,
                             download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                             shuffle=True, num_workers=2)
    elif dataset == "cifar10":
        transform = transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                             (0.2023, 0.1994, 0.2010)),])
        test_set = iCIFAR10(root='../datasets', train=False,                        
                            classes=classes,
                            download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                             shuffle=True, num_workers=2)
    elif dataset == "mnist":
        transform = transforms.Compose([#transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,),
                                                             (0.3081,)),])
        test_set = mnist(root='../datasets', train=False,
                         classes=classes, download=True,                                #####
                         transform=transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                             shuffle=True, num_workers=2)
    
    test_features, test_labels = normalFeatureReading(test_loader, model)
    
    predictions = []
    RsIn = []
    RsOut = []
    SIn = []
    SOut = []
    for test_feature, test_label in zip(test_features, test_labels):
        
        #closest_class = KNN(test_feature, (exemplar_features, exemplar_labels), K)
        R, closest_class, similarities= OSNN(test_feature, (exemplar_features, exemplar_labels), K)
        if test_label in range(num_classes):
            RsIn.append(R)
            SIn.append(similarities)
        else:
            RsOut.append(R)
            SOut.append(similarities)
        predictions.append(closest_class)
    
    trueInliers, trueInlierstrueClass, trueOutiers, falseInliers, falseOutliers = compareLabelsOSR(predictions, test_labels, range(num_classes))
    # print("Accuracy is: ", 1-compareLabels(predictions, test_labels)*1.0/len(test_labels))
    # a = [i for i, x in enumerate(Rs) if x < 1.9]
    # print("OSR", 1-len(a)*1.0 / len(test_labels))
    bins = [x*0.1 for x in range(0, 200)]
    plt.hist(RsIn, bins=bins, alpha=0.5, label="inlier LOF")
    plt.hist(RsOut, bins=bins, alpha=0.5, label="outlier LOF")
    plt.axvline(x=1.9, color="r", linestyle='--', lw=1)
    plt.legend()
    bins = [x*0.01 for x in range(0, 100)]
    plt.hist(SIn, bins=bins, alpha=0.5, label="inlier LOF")
    plt.hist(SOut, bins=bins, alpha=0.5, label="outlier LOF")
    plt.axvline(x=0.2, color="r", linestyle='--', lw=1)
    plt.legend()
    precision, recall, accuracyAll, accuracyClass = accuracy(trueInliers, trueInlierstrueClass, trueOutiers, falseInliers, falseOutliers)
    print("precision", precision)
    print("recall", recall)
    print("accuracyAll", accuracyAll)
    print("accuracyClass", accuracyClass)
    
    plt.scatter(RsIn, SIn, label="inlier LOF")
    plt.scatter(RsOut, SOut, label="outlier LOF")
    plt.legend()