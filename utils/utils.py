import os
import glob
from shutil import move
from os import rmdir
import torch
import numpy as np
import random
from torch.optim import Adam, AdamW, SGD

def val_formatter(root='data', data_dir='tiny-imagenet-200'):
    target_folder = os.path.join(root, data_dir, 'val')
    # os.mkdir(test_folder)
    val_dict = {}
    # './tiny-imagenet-200/val/val_annotations.txt'
    with open(os.path.join(target_folder, 'val_annotations.txt'), 'r') as f:
        for line in f.readlines():
            split_line = line.split('\t')
            val_dict[split_line[0]] = split_line[1]
    # './tiny-imagenet-200/val/images/*'        
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
        # target_folder + str(folder) + '/images/*'
        # target_folder + str(folder) + '/images/' + str(file)
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
        return Adam(parameters, lr, weight_decay=weight_decay)
    elif optimizer == 'adamw':
        return AdamW(parameters, lr, weight_decay=weight_decay)
    elif optimizer == 'sgd':
        return SGD(parameters, lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError

def get_target(task_class_set, y):
    task_class_set_tensor = torch.tensor(list(task_class_set))
    joint_targets = task_class_set_tensor.view(1, -1).expand(len(y), len(task_class_set_tensor))
    return joint_targets.long()

def get_ans_idx(task_class_set, y):
    ans_idx = torch.tensor([task_class_set.index(ans) for ans in y]).long()
    return ans_idx.view(len(y), 1)

def calculate_answer(energy, y_tem):
    _, pred = torch.min(energy, 1)
    train_answer = 1. * (pred == y_tem.view(-1)).sum()
    return train_answer

def calculate_answer_sbc(probs, y):
    preds = np.argmax(probs, axis=-1)
    batch_answer = np.sum(1. * (y == preds))
    return batch_answer
    
    

