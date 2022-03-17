from __future__ import print_function

import os
import sys
import argparse
import time
import math
import copy
import pickle
import numpy as np
from PIL import Image

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from torch.autograd import Variable 
import torch.nn.functional as F

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from resnet_big import SupCEResNet
from mlp import MLP
#from losses_incremental import SupConLoss
from losses import SupConLoss
from util import plot_grad_flow
from angle_similar import RKAngle
from exemplars import createExemplars

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

from data_loader import iCIFAR10, iCIFAR100, mnist, apply_transform, TinyImagenet
from dataset import customDataset


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')
    parser.add_argument('--batch_size_destill', type=int, default=10,
                        help='batch size distillation')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of training epochs')
    parser.add_argument('--num_classes', type=int, default=30,
                        help='number of training classes')
    parser.add_argument('--img_size', type=int, default=32,
                        help='image size')
    
    # incremental learning
    parser.add_argument('--incremental_training', type=bool, default=True,
                        help='if start incremental training')
    parser.add_argument('--alfa', type=float, default=0.5,
                        help='alfa to balance loss function')
    parser.add_argument('--num_init_classes', type=int, default=20,
                        help='num of old classes')
    parser.add_argument('--memory_size', type=int, default=50,
                        help='size of memory')
    parser.add_argument('--fixed_memory', type=int, default=2000,
                        help='size of fixed memory')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='100',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar10', 'cifar100', 'tinyimgnet', 'mnist', 'path'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='CE',
                        choices=['SupCon', 'SimCLR', 'CE'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.05,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save/CE/{}_models'.format(opt.dataset)
    opt.tb_path = './save/CE/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name_new = '{}_{}_class_{}_{}_lr_{}_epochs_{}_bsz_{}_temp_{}_alfa_{}_mem_{}_incremental'.\
                         format(opt.method, opt.dataset, opt.num_classes, opt.model, opt.learning_rate,
                         opt.epochs, opt.batch_size, opt.temp, opt.alfa, opt.fixed_memory)
    opt.model_name_old = '{}_{}_class_{}_{}_lr_{}_epochs_{}_bsz_{}_temp_{}_alfa_{}_mem_{}_incremental/last.pth'.\
                         format(opt.method, opt.dataset, opt.num_init_classes, opt.model, opt.learning_rate,
                         opt.epochs, opt.batch_size, opt.temp, opt.alfa, opt.fixed_memory)

    opt.exemplar_file = './exemplars/exemplar_{}_class_{}_{}_memorysize_{}_alfa_{}_temp_{}_mem_{}'.format(opt.dataset, opt.num_init_classes, 
                                                                                                          opt.model, opt.memory_size, opt.alfa, opt.temp, opt.fixed_memory)
    print(opt.exemplar_file)
    
    opt.save_folder = os.path.join(opt.model_path, opt.model_name_new)

    if not os.path.isdir(opt.save_folder):
       os.makedirs(opt.save_folder)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name_new)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name_new)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)
        
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return opt


def set_loader(opt, model_old):
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
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.std)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        #transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomApply([
        #    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        #], p=0.8),
        #transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        train_dataset = iCIFAR10(root='../datasets', train=True,
                                 classes=range(opt.num_init_classes, opt.num_classes), download=True,
                                 transform=None)
        original_dataset = iCIFAR10(root='../datasets', train=True,
                                    classes=range(0, opt.num_init_classes), download=True,
                                    transform=None)
        
    elif opt.dataset == 'cifar100':
        train_dataset = iCIFAR100(root='../datasets', train=True,
                                  classes=range(opt.num_init_classes, opt.num_classes), download=True,                 
                                  transform=None)                      
        original_dataset = iCIFAR100(root='../datasets', train=True,
                                     classes=range(0, opt.num_init_classes), download=True,
                                     transform=None)
        
    elif opt.dataset == "mnist":
        train_dataset = mnist(root='../datasets', train=True,
                              classes=range(opt.num_init_classes, opt.num_classes), download=True,
                              transform=None)
        original_dataset = mnist(root='../datasets', train=True,
                                 classes=range(0, opt.num_init_classes), 
                                 download=True, transform=None)
        
    elif opt.dataset == 'tinyimgnet':
        train_dataset = TinyImagenet(root='../datasets', train=True,
                                     classes=range(opt.num_init_classes, opt.num_classes), 
                                     download=True, transform=None)
        original_dataset = TinyImagenet(root='../datasets', train=True,
                                        classes=range(0, opt.num_init_classes), 
                                        download=True, transform=None)
        
    elif opt.dataset == 'path':
        train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                            transform=TwoCropTransform(train_transform))
    else:
        raise ValueError(opt.dataset)
        
    if os.path.isfile(opt.exemplar_file):
        with open(opt.exemplar_file, "rb") as f:
                    exemplar_sets, exemplar_labels, _, exemplar_centers = pickle.load(f)
    else:
        #transform = transforms.Compose([transforms.ToTensor(), normalize])      # TODO what kind of transform to use for exemplar selection?
        exemplar_sets, exemplar_labels, exemplar_centers = createExemplars(opt, original_dataset)
    #print(exemplar_sets.shape)
        
    exemplar_dataset = customDataset(exemplar_sets, exemplar_labels, transform=None)
    train_dataset = torch.utils.data.ConcatDataset([train_dataset, exemplar_dataset])                               
    train_dataset = apply_transform(train_dataset, train_transform)
    print(len(train_dataset))

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                                               num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader


def set_model(opt):
    model = SupCEResNet(name=opt.model, num_classes=opt.num_init_classes)
    criterion1 = torch.nn.CrossEntropyLoss()
    criterion2 = torch.nn.BCELoss()

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        #model = model.cuda()
        criterion1 = criterion1.cuda()
        criterion2 = criterion2.cuda()
        cudnn.benchmark = True

    return model, (criterion1, criterion2)


def load_model(opt, model, path):
    ckpt = torch.load(path, map_location='cpu')
    state_dict = ckpt['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v

    state_dict = new_state_dict
    model.load_state_dict(state_dict)
    model.update(opt.num_classes)
    model_new = copy.deepcopy(model)
    
    return model, model_new
    

def train(train_loader, model_old, model_new, criterion, optimizer, epoch, opt, old_targets):
    """one epoch training"""
    model_new.train()
    model_old.eval()
    model_new.cuda()
    model_old.cuda()

    criterion_cls, criterion_destill = criterion

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_cls = AverageMeter()
    losses_similar = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):

        data_time.update(time.time() - end)
        #print(images.shape)
        if torch.cuda.is_available():
             images = images.cuda(non_blocking=True)
             labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        #warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features_new = model_new(images)
        loss_cls = criterion_cls(features_new, labels)

        features_old = model_old(images)
        features_new = F.sigmoid(features_new)
        features_old = F.sigmoid(features_old)
        loss_similar = sum(criterion_destill(features_new[:,y], features_old[:,y]) for y in range(opt.num_init_classes))
    
        loss = opt.alfa * loss_cls + (1 - opt.alfa) * loss_similar

        # update metric
        losses.update(loss.item(), bsz)
        losses_cls.update(loss_cls.item(), bsz)
        losses_similar.update(loss_similar.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss_cls {losses_cls.val:.3f} ({losses_cls.avg:.3f})\t'
                  'loss_similar {losses_similar.val:.3f} ({losses_similar.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, losses_cls=losses_cls, losses_similar=losses_similar, loss=losses))
            sys.stdout.flush()

    return losses.avg



def main():
    opt = parse_option()

    # build model and criterion
    model_old, criterion = set_model(opt)      
    model_old, model_new = load_model(opt, model_old, os.path.join(opt.model_path, opt.model_name_old))
        
    # build data loader
    train_loader = set_loader(opt, model_old)               
    
    # build optimizer
    optimizer = set_optimizer(opt, model_new)
    
    epoch = 0
    save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
    save_model(model_new, optimizer, opt, epoch, save_file)
    
    old_targets = range(0, opt.num_init_classes)
    new_targets = range(opt.num_init_classes, opt.num_classes)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        #adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        #loss = incremental_train(train_loader, model, model, criterion,
        #                         old_targets, optimizer, epoch, opt)
        loss = train(train_loader, model_old, model_new, criterion, optimizer, epoch, opt, old_targets)
        
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('loss', loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model_new, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model_new, optimizer, opt, opt.epochs, save_file)
    

if __name__ == '__main__':
    main()
