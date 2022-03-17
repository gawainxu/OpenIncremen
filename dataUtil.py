#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 21:42:16 2021

@author: zhi
"""

classMap = {0: "apples", 1: "aquarium_fish", 2: "baby", 3: "bear", 4: "beaver",
            5: "bed", 6: "bee", 7: "beetle", 8: "bicycle", 9: "bottles",
            10: "bowls", 11: "boy", 12: "bridge", 13: "bus", 14: "butterfly",
            15: "camel", 16: "cans", 17: "castle", 18: "caterpillar", 19: "cattle",
            20: "chair", 21: "chimpanzee", 22: "clock", 23: "cloud", 24: "cockroach",
            25: "couch", 26: "crab", 27: "crocodile", 28: "cups", 29: "dinosaur",
            30: "dolphin", 31: "elephant", 32: "flatfish", 33: "forest", 34: "fox",
            35: "girl", 36: "hamster", 37: "house", 38: "kangaroo", 39: "keyboard", 
            40: "lamp", 41: "lawn_mower", 42: "leopard", 43: "lion", 44: "lizard",
            45: "lobster", 46: "man", 47: "maple", 48: "motorcycle", 49: "mountain",
            50: "mouse", 51: "mushrooms", 52: "oak", 53: "oranges", 54: "orchids", 
            55: "otter", 56: "palm", 57: "pears", 58: "pickup_truck", 59: "pine",
            60: "plain", 61: "plates", 62: "poppies", 63: "porcupine", 64: "possum",
            65: "rabbit", 66: "raccoon", 67: "ray", 68: "road", 69: "rocket",
            70: "roses", 71: "sea", 72: "seal", 73: "shark", 74: "shrew",
            75: "skunk", 76: "skyscraper", 77: "snail", 78: "snake", 79: "spider",
            80: "squirrel", 81: "streetcar", 82: "sunflowers", 83: "pepper", 84: "table", 
            85: "tank", 86: "telephone", 87: "television", 88: "tiger", 89: "tractor",
            90: "train", 91: "trout", 92: "tulips", 93: "turtle", 94: "wardrobe",
            95: "whale", 96: "willow", 97: "wolf", 98: "woman", 99: "worm"}

classMap = {v : k for k, v in classMap.items()}

superClasses = [["beaver", "dolphin", "otter", "seal", "whale"],
                ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
                ["orchids", "poppies", "roses", "sunflowers", "tulips"],
                ["bottles", "bowls", "cans", "cups", "plates"],
                ["apples", "mushrooms", "oranges", "pears", "peppers"],
                ["clock", "keyboard", "lamp", "telephone", "television"],
                ["bed", "chair", "couch", "table", "wardrobe"],
                ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
                ["bear", "leopard", "lion", "tiger", "wolf"],
                ["bridge", "castle", "house", "road", "skyscraper"],
                ["cloud", "forest", "mountain", "plain", "sea"],
                ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
                ["fox", "porcupine", "possum", "raccoon", "skunk"],
                ["crab", "lobster", "snail", "spider", "worm"],
                ["baby", "boy", "girl", "man", "woman"],
                ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
                ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
                ["maple", "oak", "palm", "pine", "willow"],
                ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
                ["lawn_mower", "rocket", "streetcar", "tank", "tractor"]]


def pickClass(classIdx):
    
    classNames = superClasses[classIdx]
    classList = []
    for n in classNames:
        classList.append(classMap[n])
        
    return classList