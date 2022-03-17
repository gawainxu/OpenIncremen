import torch
import torchvision.transforms as transforms
import pickle

from resnet_big import SupConResNet
from mlp import MLP
from data_loader import apply_transform
from dataset import customDataset


num_classes = 90
dataset = "cifar100"
# TODO exemplar features are different for different models !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
exemplar_file = "./exemplars/exemplar_cifar100_class_90_resnet18_memorysize_50_alfa_0.2_temp_0.05_mem_2000"

model = SupConResNet("resnet18")       # MLP()  
ckpt = torch.load("./save/SupCon_cifar100_class_90_resnet18_lr_0.001_epoch_600_bsz_512_temp_0.05_alfa_0.2_mem_2000_incremental/last.pth", map_location='cpu')
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
    labels = []

    for i, (img, label) in enumerate(data_loader):
        print(i)
         
        if loss_fcn == "supcon":
            output = model(img)
        else:
            output = model.encoder(img)
        outputs.append(output.detach().numpy())
        labels.append(label.numpy())

    with open(save_path, "wb") as f:
        pickle.dump((outputs, labels), f)
    
    
if __name__ == "__main__":
    
    loss_fcn = "supcon"
    
    with open(exemplar_file, "rb") as f:
        exemplar_sets, exemplar_labels, _, _ = pickle.load(f, encoding='latin1')
        
    if dataset == "cifar100":
         transform = transforms.Compose([#transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                                         #transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                              (0.2675, 0.2565, 0.2761)),])
    elif dataset == "mnist":
        transform = transforms.Compose([#transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                                        #transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,),
                                                            (0.3081,)),])
    exemplar_dataset = customDataset(exemplar_sets, exemplar_labels, transform=None)
    exemplar_dataset = apply_transform(exemplar_dataset, transform)
    loader = torch.utils.data.DataLoader(exemplar_dataset, batch_size=1,
                                         shuffle=True, num_workers=2)
    
    save_path = "./features/exemplar_cifar100_class_90_resnet18_memorysize_50_alfa_0.2_temp_0.05_mem_2000"
    normalFeatureReading(loader, save_path)
