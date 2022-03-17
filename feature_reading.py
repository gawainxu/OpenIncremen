#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 15:29:30 2022

@author: zhi
"""


import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
import pickle
from itertools import chain

from dataUtil import pickClass
from resnet_big import SupConResNet, SupCEResNet
from mlp import MLP
from data_loader import iCIFAR10, iCIFAR100, mnist
from ImageNet import ImageNet
from featureMerge import featureMerge

num_classes = 20
dataset = "cifar100"
mem = "2000"

model = SupConResNet("resnet18")   #SupCEResNet("resnet18", num_classes)
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


def normalFeatureReading(data_loader, save_path):
    outputs = []
    #labels = []

    for i, (img, _) in enumerate(data_loader):
        print(i)
         
        if loss_fcn == "supcon":
            output = model(img)
        else:
            output = model.encoder(img)
        outputs.append(output.detach().numpy())

    with open(save_path, "wb") as f:
        pickle.dump(outputs, f)
    
        
def reduce_exemplar_sets(exemplar_sets, m):
    
    for y, P_y in enumerate(exemplar_sets):
        exemplar_sets[y] = P_y[:m]    
    
            
def meanList(l):
    
    if len(l) == 0:
        return 0
    else:
        return sum(l)*1.0 / len(l)
        

if __name__ == "__main__":
    
    
    classes =  [i for i in range(num_classes)] #+  [i for i in range(90, 100)]  #chain(range(0, num_classes), range(95, 100))     #+ outlier_classes 
    loss_fcn = "supcon"       
    if_open = 'close'
    
    featurePaths = []
    for c in classes: 
        if dataset == "imagenet":
            transform = transforms.Compose([transforms.Resize(32),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4802, 0.4480, 0.3975), (0.2770, 0.2691, 0.2821)),])
            test_set =ImageNet(root='/home/zhi/projects/osrCL/icarl-master/datasets', train=True,
                               selectedClasses=[c],
                               transform=transform)
            loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                              shuffle=True, num_workers=2)
        elif dataset == "cifar100":
            transform = transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                                 (0.2675, 0.2565, 0.2761)),])
            test_set = iCIFAR100(root='../datasets', train=True,                        
                                 classes=[c],
                                 download=True, transform=transform)
            loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                                 shuffle=True, num_workers=2)
        elif dataset == "cifar10":
            transform = transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                 (0.2023, 0.1994, 0.2010)),])
            test_set = iCIFAR10(root='../datasets', train=True,                        
                                classes=[c],
                                download=True, transform=transform)
            loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                                 shuffle=True, num_workers=2)
        elif dataset == "mnist":
            transform = transforms.Compose([#transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                                            #transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.1307,),
                                                                 (0.3081,)),])
            test_set = mnist(root='../datasets', train=False,
                                  classes=[c], download=True,
                                  transform=transform)
            loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                                 shuffle=True, num_workers=2)
            
        save_path = "/home/zhi/projects/SupContrast-master/features/Opencifar10class4_" + str(c)
        featurePaths.append(save_path)
        normalFeatureReading(loader, save_path)
        
    featureMerge(featurePaths, "/home/zhi/projects/SupContrast-master/features/"+ if_open+'_'+loss_fcn + '_' + mem+ '_' + dataset + "class" + str(num_classes), range(num_classes))        # +"test90_100"
