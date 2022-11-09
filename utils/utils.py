import os
import glob
from shutil import move
from os import rmdir
import torch
import numpy as np
import random
from torch.optim import Adam, AdamW, SGD
from tqdm import tqdm
import torchvision.transforms as transforms

def val_formatter(root='data', data_dir='tiny-imagenet-200'):
    target_folder = os.path.join(root, data_dir, 'val')
    val_dict = {}
    with open(os.path.join(target_folder, 'val_annotations.txt'), 'r') as f:
        for line in f.readlines():
            split_line = line.split('\t')
            val_dict[split_line[0]] = split_line[1]
    paths = glob.glob(os.path.join(target_folder, 'images', '*'))
    for path in paths:
        file = path.split('/')[-1]
        folder = val_dict[file]
        if not os.path.exists(os.path.join(target_folder, str(folder))):
            os.mkdir(os.path.join(target_folder, str(folder)))
            os.mkdir(os.path.join(target_folder, str(folder) , 'images'))
    for path in paths:
        file = path.split('/')[-1]
        folder = val_dict[file]
        dest = os.path.join(target_folder, str(folder), 'images')
        move(path, dest)
        
    rmdir(os.path.join(target_folder, 'images'))

def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_optimizer(optimizer, lr, parameters, weight_decay):
    if optimizer == 'adam':
        return Adam(parameters, lr, weight_decay=weight_decay, eps=1e-04)
    elif optimizer == 'adamw':
        return AdamW(parameters, lr, weight_decay=weight_decay)
    elif optimizer == 'sgd':
        return SGD(parameters, lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise NotImplementedError

def get_target(task_class_set, y):
    task_class_set_tensor = torch.tensor(list(task_class_set))
    joint_targets = task_class_set_tensor.view(1, -1).expand(len(y), len(task_class_set_tensor))
    return joint_targets

def get_ans_idx(task_class_set, y):
    ans_idx = torch.tensor([list(task_class_set).index(ans) for ans in y]).long()
    return ans_idx.view(len(y), 1)

def calculate_answer(energy, y_tem):
    _, pred = torch.min(energy, 1)
    train_answer = 1. * (pred == y_tem.view(-1)).sum()
    return train_answer

def calculate_answer_sbc(probs, y):
    preds = np.argmax(probs, axis=-1)
    batch_answer = np.sum(1. * (y == preds))
    return batch_answer

@torch.no_grad()
def calculate_final_energy(model, device, loader, task_class_set):
    task_energy = torch.empty(0)
    answers = torch.empty(0)
    cls_energies = []
    pbar = tqdm(loader, total=loader.__len__(), position=0, leave=True)
    for sample in pbar:
        x, y = sample[0].to(device), sample[1].to(device)
        joint_targets = get_target(task_class_set, y).to(device).long()
        y_ans_idx     = get_ans_idx(task_class_set, y).to(device)
        energy = model(x, joint_targets)
        y_ans_idx = y_ans_idx.detach().cpu()
        energy_cpu = energy.detach().cpu()
        true_energy = torch.gather(energy_cpu, dim=1, index=y_ans_idx)
        task_energy = torch.cat((task_energy, true_energy))
        answers = torch.cat((answers, y_ans_idx))
    
    cur_task_class = torch.unique(answers, sorted=True)
    for cls in cur_task_class:
        idx = (answers == cls).nonzero(as_tuple=True)[0]
        cls_energies.append(task_energy[idx,:])
    return cls_energies

def aug(img, args):
    if args.learning_mode == "online":
        augmentations = transforms.Compose([
                                            transforms.RandomResizedCrop(size=(args.img_size, args.img_size), scale=(0.6, 1.)),
                                            transforms.RandomHorizontalFlip(),
                                            # transforms.RandomVerticalFlip(),
                                            # transforms.RandomRotation(degrees=(0, 30)),
                                            transforms.RandomApply([
                                                                     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                                                                    ], p=0.5)])
        if args.dataset == "splitted_mnist":
            augmentations = transforms.Compose([
                                            transforms.RandomResizedCrop(size=(args.img_size, args.img_size), scale=(0.6, 1.)),
                                            transforms.RandomHorizontalFlip(),
                                            # transforms.RandomVerticalFlip(),
                                            # transforms.RandomRotation(degrees=(0, 30)),
                                            ])
    else:
        augmentations = transforms.Compose([
                                            transforms.RandomResizedCrop(size=(args.img_size, args.img_size), scale=(0.5, 1.)),
                                            transforms.RandomHorizontalFlip(),
                                            # transforms.RandomVerticalFlip(),
                                            # transforms.RandomRotation(degrees=(0, 30)),
                                            transforms.RandomApply([
                                                                     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                                                                    ], p=0.5)])
    return augmentations(img)