# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 14:52:21 2021

@author: user
"""


import os
import matplotlib.image
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class ImageNet(Dataset):
    
    def __init__(self, root, train=True, selectedClasses=None, transform=None):

        self.dataFolder = root + "/tiny-imagenet-200"
        self.train = train
        self.transform = transform
        self.classList = sorted(os.listdir(self.dataFolder + "/train"))
        self.classMapping = {k:e for e, k in enumerate(self.classList)}
        self.trainPortion = 400
        
        dataDir = self.dataFolder + "/train"
        with open(self.dataFolder + "/words.txt", "r") as f:
            self.Allclasses = f.readlines()
        self.classes = os.listdir(dataDir)
                
        os.chdir(dataDir)
        self.dataList = []
        for folder in os.listdir(os.getcwd()):
            d = sorted(os.listdir(folder + "/images"))
            if self.train:
                d = d[:self.trainPortion]
            else:
                d = d[self.trainPortion:]
            self.dataList += [folder + "/images/" + l for l in d]
           
        data = []
        labels = []
            
        for c in range(len(self.classes)):
            if c in selectedClasses and selectedClasses is not None:
                data = data + self.get_image_class(c)
                labels += [c]*len(self.get_image_class(c))
            
        self.data = np.array(data)
        self.labels = labels
        
        assert len(data) == len(labels)
        
#        if train:
#            dataDir = self.dataFolder + "/train"
#            with open(self.dataFolder + "/words.txt", "r") as f:
#                self.Allclasses = f.readlines()
#            self.classes = os.listdir(dataDir)
#                
#            os.chdir(dataDir)
#            self.dataList = []
#            for folder in os.listdir(os.getcwd()):
#                d = os.listdir(folder + "/images")
#                self.dataList += [folder + "/images/" + l for l in d]
#            
#            train_data = []
#            train_labels = []
#            
#            for c in range(len(self.classes)):
#                if c in selectedClasses and selectedClasses is not None:
#                    train_data = train_data + self.get_image_class(c)
#                    train_labels += [c]*self.trainPortion
#            
#            self.train_data = np.array(train_data)
#            self.train_labels = train_labels
                
#        else:
#            dataDir = self.dataFolder + "/val/images"
#            os.chdir(dataDir)
#            with open(self.dataFolder + "/val/val_annotations.txt", "r") as f:
#                self.classes = f.readlines()
#                
#            self.dataList = os.listdir(dataDir)
#            self.classes = [c.split("\t")[:2] for c in self.classes]
#            self.classes = {n:c for n, c in self.classes}
#            
#            test_data = []
#            test_labels = []
#            
#            for d in self.dataList:
#                
#                #print d
#                imgName = self.classes[d]
#                #print imgName
#                label = self.classMapping[imgName]
#                if label not in selectedClasses:
#                    continue
#                #print label
#                img = matplotlib.image.imread(d)
#                if len(img.shape) != 3:
#                    img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
#                test_data.append(img)
#                test_labels.append(label)
#            
#            self.test_data = np.array(test_data)
#            self.test_labels = test_labels
        
        
    def __len__(self):
        
        return len(self.data)
        
        
        
    def __getitem__(self, idx):
        
        img, target = self.data[idx], self.labels[idx]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return idx, img, target
        
    
    def get_image_class(self, label):
        
        className = [key for key in self.classMapping.keys() if self.classMapping[key] == label][0]
        #print className
        imgs = []
        for dataName in self.dataList:
#            if self.train == False:
#                dataClass = self.classes[dataName]
#            else:
                
            dataClass = dataName                       # unintend
            if className in dataClass:
                img = matplotlib.image.imread(dataName)
                if len(img.shape) != 3:
                    img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
                imgs.append(img)
                
        return imgs