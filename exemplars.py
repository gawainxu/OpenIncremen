#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 20:55:55 2022

@author: zhi
"""

import pickle
import numpy as np
from PIL import Image

from torch.autograd import Variable 
import random


def read_features(images, model, transform):
    
    model = model.eval()
    #model.cuda()
    features = []
    f = []
    for img in images:
        #print(img.shape)
        x = Variable(transform(Image.fromarray(np.squeeze(img), mode="L")), volatile=True)
        x = x.cuda()
        feature = model(x.unsqueeze(0))
        feature = feature.cpu().data.numpy()
        f.append(feature)
        feature = feature / np.linalg.norm(feature) # Normalize
        features.append(feature[0])
        
    features = np.array(features)
    return features
    


def classExemplars_euclidean(m, images, model, transform):
 
    features = read_features(images, model, transform)
    center = centerComputing(features)
    centers = np.tile(center, (len(features), 1))
    distances = np.linalg.norm((features-centers), axis=1)

    ind = np.argsort(distances)[:m]
    exemplar_set = np.array(images)[ind]
    exemplar_features = features[ind]

    return exemplar_set, exemplar_features, center
    

def classExemplars_similar(m, images, model, transform):
    
    features = read_features(images, model, transform)
    center = centerComputing(features)
    centers = np.tile(center, (len(features), 1))
    similarities = np.matmul(features, centers.T)[:, 0]
    
    ind = np.argsort(np.abs(similarities))[:m]
    exemplar_set = np.array(images)[ind]
    exemplar_features = features[ind]

    return exemplar_set, exemplar_features, center


def classExemplars_random(m, images, model=None, transform=None):
    
    #features = read_features(images, model, transform)
    #center = centerComputing(features)
    
    ind = random.sample(range(len(images)), m)
    exemplar_set = np.array(images)[ind]
    #exemplar_features = features[ind]
    
    return exemplar_set


def createExemplars(opt, original_dataset, model_old=None, transform=None):
    
    exemplar_sets = []
    exemplar_labels = [] 
    exemplar_features_sets = []
    exemplar_centers = []
    if opt.fixed_memory == 0:
        opt.memory_per_class = opt.memory_size
    else:
        opt.memory_per_class = opt.fixed_memory // opt.num_init_classes + 1
        
    for c in range(0, opt.num_init_classes):
        print("Class: ", c)
        c_dataset = original_dataset.get_image_class(c)
        exemplar_set = classExemplars_random(int(opt.memory_per_class), c_dataset)           #  classExemplars_similar(int(opt.memory_per_class), c_dataset, model_old, transform)   #
        #exemplar_center = centerComputing(exemplar_features)
        #exemplar_centers.append(exemplar_center)
        exemplar_sets.append(exemplar_set)
        #exemplar_features_sets.append(exemplar_features)
        exemplar_labels = exemplar_labels + [c]*int(opt.memory_per_class)
        
    #exemplar_sets = np.reshape(np.array(exemplar_sets), (opt.memory_per_class*opt.num_init_classes, 1, opt.img_size, opt.img_size))          #### 1
    exemplar_sets = np.reshape(np.array(exemplar_sets), (opt.memory_per_class*opt.num_init_classes, opt.img_size, opt.img_size, 3)) 
    exemplar_labels = np.squeeze(np.array(exemplar_labels))   
    
    with open(opt.exemplar_file, "wb") as f:
        pickle.dump((exemplar_sets, exemplar_labels, exemplar_features_sets, exemplar_centers), f)
    
    return exemplar_sets, exemplar_labels, exemplar_centers


def centerComputing(features):
    
    return np.mean(features, 0)


