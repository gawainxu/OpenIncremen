#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 10:17:58 2021

@author: zhi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class RKAngle(nn.Module):
    def forward(self, last, current, label, old_targets):
        
        """
        Is it the similarity for all the features ???
        """
        bsz = label.shape[0]
        label = label.cpu().numpy()
        m = []
        for i, l in enumerate(label):
            if l in old_targets:
                m.append(i)
                
        #print("mask is:", m)
        
        last = last[m, :]
        current = current[m, :]
        
        with torch.no_grad():
            td = (last.unsqueeze(0) - last.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2))
            t_angle = t_angle.view(-1)
            
            #print("t_angle", t_angle)
            
        sd = (current.unsqueeze(0) - current.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)
        
        #print("s_angle", s_angle)
        
        loss = F.smooth_l1_loss(s_angle, t_angle, reduction="elementwise_mean")
                
        if len(m) == 0:
            loss[loss != loss] = 0
            
        #print("similar loss", loss)
        
        return loss