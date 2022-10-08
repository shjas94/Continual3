import os
import torch
import numpy as np
import random
from torch.optim import Adam, AdamW, SGD


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_optimizer(optimizer, lr, parameters):
    if optimizer == 'adam':
        return Adam(parameters, lr)
    elif optimizer == 'adamw':
        return AdamW(parameters, lr)
    elif optimizer == 'sgd':
        return SGD(parameters, lr)
    else:
        raise NotImplementedError

def get_target(task_class_set, y):
    task_class_set_tensor = torch.tensor(list(task_class_set))
    joint_targets = task_class_set_tensor.view(1, -1).expand(len(y), len(task_class_set_tensor))
    return joint_targets.long()

def calculate_answer(energy, y_tem):
    _, pred = torch.min(energy, 1)
    train_answer = 1. * (pred == y_tem.view(-1)).sum()
    return train_answer

def calculate_answer_sbc(probs, y):
    preds = np.argmax(probs, axis=-1)
    batch_answer = np.sum(1. * (y == preds))
    return batch_answer
    
    

