# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 18:40:28 2021

@author: user
"""

import pickle

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#import libmr

"""
This file implements the OSR methods based on the feature maps extracted by ICARL
after incremental learning
"""

def sortFeatures(mixedFeatures, labels, numClasses):
        
    sortedFeatures = []
    for i in range(numClasses):
        sortedFeatures.append([])
    
    for i, l in enumerate(labels):
        l = l.item()                          # remove when labels are not tensors/array
        feature = mixedFeatures[i]
        feature = feature.reshape([-1])
        sortedFeatures[l].append(feature)
        
    # Attention the #samples for each class are different
    return sortedFeatures
    
    
def EuclideanStat(inlierFeatures):
    
    means = []
    for cFeatures in inlierFeatures:
        mean = np.mean(cFeatures, axis=0)
        means.append(mean)
        
    return means

    
def closestClass(sampleFeature, classCenters):
    
    distances = []
    for c in classCenters:
        distances.append(np.linalg.norm(c-sampleFeature))
        
    closestClass = np.argmin(np.squeeze(np.array(distances)))
    smallestDistance = np.min(np.squeeze(np.array(distances)))
    
    return closestClass, smallestDistance
    


def sortDistances(minDistances, mixLabels, num_classes):
    
    sortedDistances = [[] for i in range(num_classes+1)]
    for d, l in zip(minDistances, mixLabels):
        if l >= num_classes:
            sortedDistances[-1].append(d)
        else:
            sortedDistances[l.item()].append(d)


def EuclideanDistance(feature1, feature2):
    
    return np.linalg.norm((feature1 - feature2))



def interClassSeperation(means):
    
    """
    It is to compute the distances 
    """
    distances = []
    for i in range(0, len(means)):
        for j in range(i+1, len(means)):
            distances.append(EuclideanDistance(means[i], means[j]))
            
    return np.min(np.array(distances))


def intraClassSeperation(sortedFeatures, classCenters):
    
    num_instances  = 0
    sum_distances = 0
    for features, center in zip(sortedFeatures, classCenters):
        num_instances += len(features)
        for feature in features:
            sum_distances += EuclideanDistance(feature, center)
            
    return sum_distances / num_instances



if __name__ == "__main__":
    
    """
    OSR based on EVT
    """
    
    # Load the features of the inliers of each class
    numClasses = 20
    featureInliersPath = "./features/Opencifar100_exemplar20"    
    
    with open(featureInliersPath, "rb") as f:
        featuresInliers, labelsInliers = pickle.load(f)
    
    sortedFeatures = sortFeatures(featuresInliers, labelsInliers, numClasses)
    means = EuclideanStat(sortedFeatures)    
    
    print("Intra Seperation: ", intraClassSeperation(sortedFeatures, means))
    print("Inter Seperation: ", interClassSeperation(means))
    print("Ratio: ", intraClassSeperation(sortedFeatures, means)/interClassSeperation(means))
    
    # IntraCe = []
    # InterCe = []
    # InterContra = []
    # IntraContra = []
    
    # Ce = []
    # Contra = []
    
    # crange = np.arange(10, 80, 10)
    
    # for numClasses in crange:
    #     featureContraPath = "../features/contra_cifar100class" + str(numClasses)    
    #     featureCePath = "../features/ce_cifar100class" + str(numClasses)
        
    #     with open(featureContraPath, "rb") as f:
    #         featuresContra, labelsContra = pickle.load(f)
            
    #     sortedFeaturesContra = sortFeatures(featuresContra, labelsContra, numClasses)
    #     meansContra = EuclideanStat(sortedFeaturesContra)   
            
    #     with open(featureCePath, "rb") as f:
    #         featuresCe, labelsCe = pickle.load(f)
            
    #     sortedFeaturesCe = sortFeatures(featuresCe, labelsCe, numClasses)
    #     meansCe = EuclideanStat(sortedFeaturesCe)  
        
    #     IntraCe.append(intraClassSeperation(sortedFeaturesCe, meansCe))
    #     IntraContra.append(intraClassSeperation(sortedFeaturesContra, meansContra))
    #     InterCe.append(interClassSeperation(meansCe))
    #     InterContra.append(interClassSeperation(meansContra))
        
    #     Ce.append(intraClassSeperation(sortedFeaturesCe, meansCe) / interClassSeperation(meansCe))
    #     Contra.append(intraClassSeperation(sortedFeaturesContra, meansContra) / interClassSeperation(meansContra))
        
    # width = 2.0
    # plt.bar(crange-1, Ce, width)
    # plt.bar(crange+1, Contra, width)
    # plt.xlabel("Classes", fontsize=15)
    # plt.ylabel('$R_{s}$', fontsize=15)
    # plt.legend(["Corss Entropy", "Contrastive Loss"])
        
    
    
    # fit the EVT model
    
#    intraClassDistancesTrain = []
#    mrS = []
#    tailSize = 50
#    
#    for c in range(numClasses):
#        mean = means[c]
#        classFeatures  = sortedFeatures[c]
#        for feature in classFeatures:
#            d = EuclideanDistance(mean, feature)
#            intraClassDistancesTrain.append(d)
#        
#        mr = libmr.MR() 
#        mr.fit_high(np.array(intraClassDistancesTrain), tailSize)
#        mrS.append(mr)
#    
#    
#    #load the testing features and test
#    
#    featureTestPath = "../features/cifar10class2Test6_10"   
#    threshold = 0.9
#    labels = []
#    intraClassDistancesTest = []
#    interClassDistancesTest = []
#    
#    
#    with open(featureTestPath, "rb") as f:
#        featuresTest, labelsTest = pickle.load(f)
#        
#    for feature, label in zip(featuresTest, labelsTest):
#        distances = []
#        for c in range(numClasses):
#            mean = means[c]
#            distance = EuclideanDistance(mean, feature)
#            distances.append(distance)
##            
##            mr = mrS[c]
##            p = mr.cdf(distance)
##            
##            if p >= threshold:
##                labels.append(1000)
##            else:
##                labels.append(label)
#        mindistance = np.min(np.array(distances))
#        if label >= numClasses:
#            interClassDistancesTest.append(mindistance)
#        else:
#            intraClassDistancesTest.append(mindistance)
#            
#    bins = [x*1e-1 for x in range(0, 1000)] #[x*1e-3 for x in range(0, 1000)]
#    plt.hist(intraClassDistancesTrain, bins=bins, alpha=0.5, label="inlier Test")
#    plt.hist(interClassDistancesTest, bins=bins, alpha=0.5, label="outlier Test")
#    plt.legend(loc='upper right')