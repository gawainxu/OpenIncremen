#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 16:00:10 2021

@author: zhi
"""

import json
import pickle
import numpy as np


def featureMerge(featureList, savePath, inlierRange):
    
    featureMaps = []
    labels = []
    for featurePath in featureList:
        
        c = int(featurePath.split("_")[-1])
        with open(featurePath, "rb") as f:
            featureMap = pickle.load(f)
        featureMaps = featureMaps + featureMap
        if c in inlierRange:
            labels = labels + [c] * len(featureMap)
        else:
            labels = labels + [1000] * len(featureMap)
        
    featureMaps = np.squeeze(np.array(featureMaps))
    labels = np.squeeze(np.array(labels))
    
    with open(savePath, 'wb') as f:
        pickle.dump((featureMaps, labels), f)
        
