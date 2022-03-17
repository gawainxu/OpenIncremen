from __future__ import print_function

import os
import sys
import argparse
import time
import math

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

import numpy as np
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer
from resnet_big import SupConResNet, LinearClassifier
from data_loader import iCIFAR10, iCIFAR100, mnist, TinyImagenet

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--num_classes', type=int, default=100,
                        help='number of training classes')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar10', 'cifar100'], help='dataset')
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_path = './save/'
    opt.model_name = '{}_{}_class_{}_{}_lr_0.001_epoch_600_bsz_512_temp_0.05_alfa_0.2_mem_2000_incremental/last.pth'.\
                         format(opt.method, opt.dataset, opt.num_classes, opt.model)
    
    opt.open_path = "./save/Linear_{}_{}_class_{}_mem_2000.pt".format(opt.method, opt.dataset, opt.num_classes)

    return opt


def set_model(opt):
    model = SupConResNet(name=opt.model)
    criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name=opt.model, num_classes=opt.num_classes)
    classifier.load_state_dict(torch.load(opt.open_path))

    ckpt = torch.load(os.path.join(opt.model_path, opt.model_name), map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = model.encoder #torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

    return model, classifier, criterion


def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'tinyimgnet':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif opt.dataset == 'mnist':
        mean = (0.1307,)
        std = (0.3081,)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        #transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        test_dataset = iCIFAR10(root='../datasets', train=False,
                                 classes=range(opt.num_classes), download=True,
                                 transform=train_transform)
        
    elif opt.dataset == 'cifar100':
        test_dataset = iCIFAR100(root='../datasets', train=False,
                                  classes=range(opt.num_classes), download=True,
                                  transform=train_transform)
    elif opt.dataset == "mnist":
        test_dataset = mnist(root='../datasets', train=False,
                              classes=range(opt.num_classes), download=True,
                              transform=train_transform)
        
    elif opt.dataset == 'tinyimgnet':
        test_dataset = TinyImagenet(root='../datasets', train=False,
                                     classes=range(opt.num_classes), download=True,
                                     transform=train_transform)
    else:
        raise ValueError(opt.dataset)

    test_sampler = None
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, 
                                              num_workers=opt.num_workers, 
                                              pin_memory=True, sampler=test_sampler)
    return test_loader


def accuracy(outputs, targets):
    
    unequ = 0
    for pred, target in zip(outputs, targets):
        
        pred = np.argmax(pred)
        if pred != target:
                unequ += 1
        
    return 1 - unequ*1.0 / len(outputs)


def test(test_loader, model, classifier, opt):
    """validation"""
    model.eval()
    classifier.eval()

    with torch.no_grad():
        preds = []
        targets = []
        for idx, (images, labels) in enumerate(test_loader):
            images = images.float().cuda()
            #labels = labels.cuda()

            # forward
            output = classifier(model.encoder(images))
            preds.append(np.squeeze(output.cpu().detach().numpy()))
            targets.append(labels.numpy())

    print(accuracy(preds, targets))


def main():

    opt = parse_option()

    # build data loader
    test_loader = set_loader(opt)

    # build model and criterion
    model, classifier, criterion = set_model(opt)

    # build optimizer
    test(test_loader, model, classifier, opt)
        


if __name__ == '__main__':
    main()
