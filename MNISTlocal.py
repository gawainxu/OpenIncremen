from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import codecs
import gzip
import struct


"""
Data Preparation for MNIST Handwriting dataset used for pytorch training
"""

class MNISTlocal(data.Dataset):


      datas = ['train-images-idx3-ubyte.gz',
               'train-labels-idx1-ubyte.gz',
               't10k-images-idx3-ubyte.gz',
               't10k-labels-idx1-ubyte.gz']

      taining_file = 'training.pt'
      test_file = 'test.pt'

      def __init__(self, root, train=True, transform=None, target_transform=None):
   
          self.root = os.path.expanduser(root)
          self.train = train
          self.transform = transform
          self.target_transform = target_transform

          if self.train:
             self.train_data = read_img_file(root + '/mnist/raw/train-images-idx3-ubyte.gz')
             self.train_labels = read_label_file(root + '/mnist/raw/train-labels-idx1-ubyte.gz')
          else:
             self.test_data =  read_img_file(root + '/mnist/raw/t10k-images-idx3-ubyte.gz')
             self.test_labels =  read_label_file(root + '/mnist/raw/t10k-labels-idx1-ubyte.gz')               

       
      def __getitem__(self, index):
          """
          Args:
              index(int): Index
          Returns:
              tuple: (image, target) where target is index of the target class
          """
          if self.train:
             img, target = self.train_data[index], self.train_labels[index]
          else:
             img, target = self.test_data[index], self.test_labels[index]

          if self.transform is not None:
             img = self.transform(img)

          if self.target_transform is not None:
             target = self.target_transform(target)

          sample = {"image": img, "target": target}

          return sample


      def __len__(self):
          if self.train:
             return len(self.train_data)
          else:
             return len(self.test_data) 




def read_label_file(path):
    # read all images at once  
    # return: torch tensor  

    f = gzip.open(path, 'rb')
    f.read(8)                  # 4 byte integer magic number, 4 byte number of items
    num_imgs = 10000        #struct.unpack('>I', f.read(4))[0]  !!!!!!!!!!!
    
    buf = f.read(num_imgs)
    label = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    label_torch = torch.from_numpy(label).view(num_imgs).long()

    return label_torch


def read_img_file(path):
    # read all images at once
    # return: torch tensor

    f = gzip.open(path, 'rb')
    f.read(8)                                         # skip the first 16 bytes, 4 byte integer magic number, 
    num_imgs = 10000    #struct.unpack('>I', f.read(4))[0]       # 4 byte integer number of images, !!!!!!!!!!!!!!!                   
    num_rows = struct.unpack('>I', f.read(4))[0]      # 4 byte number of rows
    num_cols = struct.unpack('>I', f.read(4))[0]      # 4 byte number of columns
    buf = f.read(num_cols*num_rows*num_imgs)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    #data.reshape(num_images, image_size, image_size, 1)
    data_torch = torch.from_numpy(data).view(num_imgs, 1, num_rows, num_cols)
    return data_torch
         

if __name__ == "__main__":

     root_path = "../datasets"
     dataset = MNISTlocal(root_path, train=False)
     train_loader = torch.utils,data.DataLoader(dataset)
