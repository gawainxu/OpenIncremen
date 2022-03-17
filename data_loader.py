from torchvision.datasets import CIFAR10, CIFAR100, MNIST
import dataUtil
import numpy as np
import os
import gzip
import torch
import struct
from PIL import Image
import torchvision.transforms as transforms
from util import TwoCropTransform
from torch.utils.data import Dataset
import torch.utils.data as data


class iCIFAR10(CIFAR10):
    def __init__(self, root,
                 classes=range(10),
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False):
        super(iCIFAR10, self).__init__(root,
                                       train=train,
                                       transform=transform,
                                       target_transform=target_transform,
                                       download=download)

        # Select subset of classes
        if self.train:
            train_data = []
            train_labels = []

            for i in range(len(self.data)):
                if self.targets[i] in classes:
                    train_data.append(self.data[i])
                    train_labels.append(self.targets[i])

            self.train_data = np.array(train_data)
            self.train_labels = train_labels

        else:
            test_data = []
            test_labels = []

            for i in range(len(self.data)):
                if self.targets[i] in classes:
                    test_data.append(self.data[i])
                    test_labels.append(self.targets[i])

            self.test_data = np.array(test_data)
            self.test_labels = test_labels

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def get_image_class(self, label):
        return self.train_data[np.array(self.train_labels) == label]
    
    
    def get_part_data(self, xidxs):
        
        self.train_data = np.delete(self.train_data, xidxs, 0)
        self.train_labels = np.delete(self.train_labels, xidxs, 0)


    def append(self, images, labels):
        """Append dataset with images and labels

        Args:
            images: Tensor of shape (N, C, H, W)
            labels: list of labels
        """

        self.train_data = np.concatenate((self.train_data, images), axis=0)
        self.train_labels = self.train_labels + labels



class iCIFAR100(CIFAR100):
    def __init__(self, root,
                 classes=range(100),
                 superClass = None,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False):
        super(iCIFAR100, self).__init__(root,
                                        train=train,
                                        transform=transform,
                                        target_transform=target_transform,
                                        download=download)
        
        if superClass is not None:
            classes = [dataUtil.classMap[n] for n in dataUtil.superClasses[superClass]] 

        # Select subset of classes
        if self.train:
            train_data = []
            train_labels = []

            for i in range(len(self.data)):
                if self.targets[i] in classes:
                    train_data.append(self.data[i])
                    train_labels.append(self.targets[i])

            self.train_data = np.array(train_data)
            self.train_labels = train_labels
            
        else:
            test_data = []
            test_labels = []

            for i in range(len(self.data)):
                if self.targets[i] in classes:
                    test_data.append(self.data[i])
                    test_labels.append(self.targets[i])

            self.test_data = np.array(test_data)
            self.test_labels = test_labels

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def get_image_class(self, label):
        return self.train_data[np.array(self.train_labels) == label]
    
    
    def get_part_data(self, xidxs):
        
        self.train_data = np.delete(self.train_data, xidxs, 0)
        self.train_labels = np.delete(self.train_labels, xidxs, 0)


    def append(self, images, labels):
        """Append dataset with images and labels

        Args:
            images: Tensor of shape (N, C, H, W)
            labels: list of labels
        """

        self.train_data = np.concatenate((self.train_data, images), axis=0)
        self.train_labels = self.train_labels + labels
        
        
class mnist(MNIST):
    
    def __init__(self, root,
                 classes=range(10),
                 train = True,
                 transform = None,
                 target_transform = None,
                 download = True):
        super(mnist, self).__init__(root, train=train,
                                    transform=transform,
                                    target_transform=target_transform,
                                    download=download)
        
        # Select subset of classes
        if self.train:
            train_data = []
            train_labels = []

            for i in range(0, len(self.data), 10):
                if self.targets[i] in classes:
                    train_data.append(self.data[i])
                    train_labels.append(self.targets[i])

            self.traindata = torch.stack(train_data).numpy()
            self.trainlabels = train_labels

        else:
            test_data = []
            test_labels = []

            for i in range(0, len(self.data), 10):
                if self.targets[i] in classes:
                    test_data.append(self.data[i])
                    test_labels.append(self.targets[i])   # it is torch tensor !!!!!!!!!!!!!

            print(len(test_data))
            self.testdata = torch.stack(test_data).numpy()
            self.testlabels = test_labels
        
        
    def __getitem__(self, index):
        if self.train:
            img, target = self.traindata[index], self.trainlabels[index]
        else:
            img, target = self.testdata[index], self.testlabels[index]

        img = Image.fromarray(img, mode='L')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    
    
    def __len__(self):
        if self.train:
            return len(self.traindata)
        else:
            return len(self.testdata)
        
        
    def get_image_class(self, label):
        return self.traindata[np.array(self.trainlabels) == label]
    
    
class mnist1(data.Dataset):


      datas = ['train-images-idx3-ubyte.gz',
               'train-labels-idx1-ubyte.gz',
               't10k-images-idx3-ubyte.gz',
               't10k-labels-idx1-ubyte.gz']

      taining_file = 'training.pt'
      test_file = 'test.pt'

      def __init__(self, root, classes=range(10), train=True, download=False,
                   transform=None, target_transform=None):
   
          self.root = os.path.expanduser(root)
          self.train = train
          self.transform = transform
          self.target_transform = target_transform

          if self.train:
             self.train_data = self.read_img_file(root + '/mnist/raw/train-images-idx3-ubyte.gz')
             self.train_labels = self.read_label_file(root + '/mnist/raw/train-labels-idx1-ubyte.gz')
             
             train_data = []
             train_labels = []

             for i in range(0, len(self.train_data)):
                 if self.train_labels[i] in classes:
                     train_data.append(self.train_data[i])
                     train_labels.append(self.train_labels[i])

             self.train_data = torch.stack(train_data).numpy()
             self.train_labels = train_labels
             
          else:
             self.test_data =  self.read_img_file(root + '/mnist/raw/train-images-idx3-ubyte.gz')
             self.test_labels =  self.read_label_file(root + '/mnist/raw/train-labels-idx1-ubyte.gz')             
             
             test_data = []
             test_labels = []

             for i in range(len(self.test_data), 10):
                 if self.test_labels[i] in classes:
                     test_data.append(self.test_data[i])
                     test_labels.append(self.test_labels[i])   # it is torch tensor !!!!!!!!!!!!!

             print(len(test_data))
             self.test_data = torch.stack(test_data).numpy()
             self.test_labels = test_labels

       
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
             img = Image.fromarray(np.squeeze(img), mode='L')
             img = self.transform(img)

          if self.target_transform is not None:
             target = self.target_transform(target)

          return img, target


      def __len__(self):
          if self.train:
             return len(self.train_data)
          else:
             return len(self.test_data) 
         
      def get_image_class(self, label):
          if self.train:
              return self.train_data[np.array(self.train_labels) == label]
          else:
              return self.test_data[np.array(self.test_labels) == label]
         
            
      def read_label_file(self, path):
          # read all images at once  
          # return: torch tensor  
          
          f = gzip.open(path, 'rb')
          f.read(8)                  # 4 byte integer magic number, 4 byte number of items
          if self.train:
              num_imgs = 10000        #struct.unpack('>I', f.read(4))[0]  !!!!!!!!!!!
          else:
              num_imgs = 10000
    
          buf = f.read(num_imgs)
          label = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
          label_torch = torch.from_numpy(label).view(num_imgs).long()

          return label_torch
      
        
      def read_img_file(self, path):
          # read all images at once
          # return: torch tensor
          f = gzip.open(path, 'rb')
          f.read(8)                                         # skip the first 16 bytes, 4 byte integer magic number, 
          if self.train:
              num_imgs = 10000                                  #struct.unpack('>I', f.read(4))[0]       # 4 byte integer number of images, !!!!!!!!!!!!!!! 
          else:
              num_imgs = 10000                  
          num_rows = struct.unpack('>I', f.read(4))[0]      # 4 byte number of rows
          num_cols = struct.unpack('>I', f.read(4))[0]      # 4 byte number of columns
          buf = f.read(num_cols*num_rows*num_imgs)
          data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
          #data.reshape(num_images, image_size, image_size, 1)
          data_torch = torch.from_numpy(data).view(num_imgs, 1, num_rows, num_cols)
          return data_torch
                
    

class TinyImagenet(Dataset):
    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """
    def __init__(self, root, classes=range(200), train=True, transform=None,
                target_transform=None, download=False):
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        if download:
            if os.path.isdir(root) and len(os.listdir(root)) > 0:
                print('Download not needed, files already on disk.')
            else:
                from google_drive_downloader import GoogleDriveDownloader as gdd

                # https://drive.google.com/file/d/1Sy3ScMBr0F4se8VZ6TAwDYF-nNGAAdxj/view
                print('Downloading dataset')
                gdd.download_file_from_google_drive(
                    file_id='1Sy3ScMBr0F4se8VZ6TAwDYF-nNGAAdxj',

                    dest_path=os.path.join(root, 'tiny-imagenet-processed.zip'),
                    unzip=True)

        self.data = []
        for num in range(10):                                       #20
            self.data.append(np.load(os.path.join(
                root, 'processed/x_%s_%02d.npy' %
                      ('train' if self.train else 'val', num+1))))
        self.data = np.concatenate(np.array(self.data))

        self.targets = []
        for num in range(10):                               # 20
            self.targets.append(np.load(os.path.join(
                root, 'processed/y_%s_%02d.npy' %
                      ('train' if self.train else 'val', num+1))))
        self.targets = np.concatenate(np.array(self.targets))
        
        train_data = []
        train_labels = []

        for i in range(len(self.data)):
            if self.targets[i] in classes:
                train_data.append(self.data[i])
                train_labels.append(self.targets[i])

        self.data = np.array(train_data)
        self.targets = train_labels


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.uint8(255 * img))                     ## put it in non transform ????????????
        original_img = img.copy()

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, original_img, self.logits[index]

        return img, target
    
    def get_image_class(self, label):

        return self.data[np.array(self.targets) == label]



class apply_transform(Dataset):
    
    def __init__(self, init_dataset, transform):
        self.init_dataset = init_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.init_dataset)
    
    def __getitem__(self, idx):
        img, target = self.init_dataset[idx]
        # if not isinstance(img, np.ndarray):
        #    img = img.numpy()
        #     img = np.repeat(np.expand_dims(img, 2), 3, axis=2)
        # print(img.shape)
        # img = Image.fromarray(img, mode="L")
        
        if self.transform is not None:
            img = Image.fromarray(np.squeeze(img))                           ###########  , mode="L"
            img = self.transform(img)
            
        return img, target
        



if __name__ == "__main__":
    transform = transforms.Compose([
       # transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])                                      # (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    # train_set = iCIFAR100(root='../datasets/', train=True,
    #                        classes=range(0, 10),
    #                        download=False, transform=None)
    # train_set = apply_transform(train_set, TwoCropTransform(transform))
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=200, shuffle=True, num_workers=1)
    # for i, (img, l) in enumerate(train_loader):
    #     if i == 0:
    #         break
    root_path = "../datasets"
    dataset = mnist(root_path, train=True)
    b = dataset.get_image_class(0)
    # train_set = apply_transform(dataset, TwoCropTransform(transform))
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True,
    #                                            num_workers=2, pin_memory=True)
    #dataset = TinyImagenet(root_path, classes=[1,2])
    # a = dataset.get_image_class(2)
    # dataset = apply_transform(dataset, TwoCropTransform(transform))