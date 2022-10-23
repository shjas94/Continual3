
import os
import yaml
from tqdm import tqdm
import torch
from torch.utils.data import ConcatDataset
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, energy_distance
from modules.dataset import Coreset_Dataset
from utils.utils import get_target, get_ans_idx
# main 함수에서 loader 별로 for문 돌기 전에 선언
class Offline_Coreset_Manager(object):
    def __init__(self, args, num_classes_per_tasks, num_memory_per_class, available_memory_options):
        self.args = args    
        self.coreset = Coreset_Dataset(torch.empty(0), torch.empty(0))
        self.num_classes_per_tasks = num_classes_per_tasks
        self.num_memory_per_tasks = num_memory_per_class
        self.classes = self._get_class_name()
        self.wasserstein_dist_df = pd.DataFrame(index=available_memory_options, columns=self.classes)
        self.energy_dist_df = pd.DataFrame(index=available_memory_options, columns=self.classes)
        self.memory_energy = []
                
    def add_coreset(self, model, device, memory, augmentation=None):
        x, y, energy = memory
        cl_list = sorted(list(set(list(y.numpy()))))
        
        for cur_class, cl in tqdm(enumerate(cl_list[-1*self.num_classes_per_tasks:]), desc="Getting Coreset"):
            index = (y == torch.tensor(cl)).nonzero(as_tuple=True)
            cur_x = x[index]
            cur_y = y[index]
            cur_energy = energy[index]

            if self.args.memory_option == "random_sample":
                self.memory_energy.append(self._random_sample(cur_x, cur_y, cur_energy))
                
            elif self.args.memory_option == "min_score":
                self.memory_energy.append(self._score_based(model, device, cur_x, cur_y, cur_energy, False))
                
            elif self.args.memory_option == "max_score":
                self.memory_energy.append(self._score_based(model, device, cur_x, cur_y, cur_energy, True))
                
            elif self.args.memory_option == "low_energy":
                self.memory_energy.append(self._energy_based(cur_x, cur_y, cur_energy, False))
                
            elif self.args.memory_option == "high_energy":
                self.memory_energy.append(self._energy_based(cur_x, cur_y, cur_energy, True))
                
            elif self.args.memory_option == "confused_pred":
                self.memory_energy.append(self._confused_pred(cur_x, cur_y, cur_energy))
                
            elif self.args.memory_option == "bin_based":
                self.memory_energy.append(self._bin_based(cur_x, cur_y, cur_energy))
                
            elif self.args.memory_option == "representation_based":
                '''
                Not Implemented yet
                '''
                # return self._representation_based()
                raise NotImplementedError("Not Implemented yet")
            elif self.args.memory_option == "ensemble":
                '''
                Not Implemented yet
                '''
                # return self._ensemble()
                raise NotImplementedError("Not Implemented yet")
            else:
                raise NotImplementedError("Not a valid option for coreset selection")
            self.wasserstein_dist_df[self.classes[cur_class]][self.args.memory_option] = self._calculate_wasserstein_distance(cur_energy, self.memory_energy[cur_class])
            self.energy_dist_df[self.classes[cur_class]][self.args.memory_option] = self._calulate_energy_distance(cur_energy, self.memory_energy[cur_class])
    
    def _get_class_name(self):
        with open(os.path.join('utils','labels.yml')) as outfile:
            label_map = yaml.safe_load(outfile)
        if self.args.dataset == 'cifar10':
            data_label_map = label_map['CIFAR10_LABELS']
        elif self.args.dataset == 'tiny_imagenet':
            data_label_map = label_map['TINYIMAGENET_LABELS']
        else:
            raise NotImplementedError('Only Available for Cifar10..... others will be implemented soon.....')
        cifar_labels = [i for i in range(self.args.num_classes)]
        return [list(data_label_map.keys())[list(data_label_map.values()).index(l)]\
                for l in cifar_labels]
        
    def _random_sample(self, cur_x, cur_y, cur_energy, augmentation=None):
        memory_energy = torch.empty(0)
        idx = torch.randperm(len(cur_x))[:self.args.memory_size]
        memory_x = cur_x[idx]
        memory_y = cur_y[idx]
        mem_energy = cur_energy[idx]
        memory_energy = torch.cat((memory_energy, mem_energy.detach().cpu().view(-1)))
        memory_dataset = Coreset_Dataset(memory_x, memory_y, transform=augmentation)
        self.coreset = ConcatDataset([self.coreset, memory_dataset])
        return memory_energy
    
    def _energy_based(self, cur_x, cur_y, cur_energy, energy_mode=True, augmentation=None):
        '''
        energy_mode = True if you select coreset by high_energy
        else False
        '''
        memory_energy = torch.empty(0)
        idx = torch.topk(cur_energy, self.args.memory_size, dim=0, largest=energy_mode)[1]
        memory_x = cur_x[idx].view((-1, self.args.num_channels, self.args.img_size, self.args.img_size))
        memory_y = cur_y[idx]
        memory_dataset = Coreset_Dataset(memory_x, memory_y, transform=augmentation)
        mem_energy = cur_energy[idx]
        memory_energy = torch.cat((memory_energy, mem_energy.detach().cpu().view(-1)))
        self.coreset = ConcatDataset([self.coreset, memory_dataset])
        return memory_energy
    
    def _calculate_score(self, model, device, cur_x, cur_y):
        score_tensor = torch.empty(0)
        for i in range(len(cur_x)):
            tmp_x = cur_x[i,:,:,:].view((-1, self.args.num_channels, self.args.img_size, self.args.img_size)).to(device)
            tmp_y = cur_y[i].view(-1).long().to(device)
            tmp_x.requires_grad = True
            e = model(tmp_x, tmp_y)[0]
            e.backward()
            score = tmp_x.grad
            sum = torch.abs(torch.mean(score)) + torch.var(score)
            score_tensor = torch.cat((score_tensor, sum.detach().cpu().view(-1)))
        return score_tensor
    
    def _score_based(self, model, device, cur_x, cur_y, cur_energy, score_mode=True, augmentation=None):
        '''
        score_mode = True if you select coreset by max_score
        else False
        '''
        memory_energy = torch.empty(0)
        score_tensor = self._calculate_score(self.args, model, device, cur_x, cur_y)
        idx = torch.topk(score_tensor, self.args.memory_size, dim=0, largest=score_mode)[1]
        memory_x = cur_x[idx]
        memory_y = cur_y[idx]
        mem_energy = cur_energy[idx]
        memory_energy = torch.cat((memory_energy, mem_energy.detach().cpu().view(-1)))
        memory_dataset = Coreset_Dataset(memory_x, memory_y, transform=augmentation)
        self.coreset = ConcatDataset([self.coreset, memory_dataset])
        return memory_energy
    
    def _confused_pred(self, cur_x, cur_y, cur_energy, augmentation=None):
        memory_energy = torch.empty(0)
        idx = torch.topk(cur_energy, self.args.memory_size, dim=0, largest=True)[1]
        memory_x = cur_x[idx].view(self.args.img_size)
        memory_y = cur_y[idx]
        mem_energy = cur_energy[idx]
        memory_energy = torch.cat((memory_energy, mem_energy.detach().cpu().view(-1)))
        memory_dataset = Coreset_Dataset(memory_x, memory_y, transform=augmentation)
        self.coreset = ConcatDataset([self.coreset, memory_dataset])
        return memory_energy
    
    def _bin_based(self, cur_x, cur_y, cur_energy, augmentation=None):
        memory_energy = torch.empty(0)
        flatten_energy = cur_energy.view(-1) # |C_t|
        _, bin_idx = torch.sort(flatten_energy)
        bins = torch.linspace(0, len(bin_idx), self.args.memory_size).long()
        bins[-1] = bins[-1]-1
        memory_x = cur_x[bin_idx][bins]
        memory_y = cur_y[bin_idx][bins]
        mem_energy = cur_energy[bin_idx][bins]
        memory_energy = torch.cat((memory_energy, mem_energy.detach().cpu().view(-1)))
        memory_dataset = Coreset_Dataset(memory_x, memory_y, transform=augmentation)
        self.coreset = ConcatDataset([self.coreset, memory_dataset])
        return memory_energy
    
    def _representation_based(self):
        pass
    
    def _ensemble(self):
        pass
    
    def _calculate_wasserstein_distance(self, cur_energy, memory_energy):
        return np.round(wasserstein_distance(cur_energy.cpu().numpy().flatten(), memory_energy.cpu().numpy().flatten()), 5)
    
    def _calulate_energy_distance(self, cur_energy, memory_energy):
        return np.round(energy_distance(cur_energy.cpu().numpy().flatten(), memory_energy.cpu().numpy().flatten()), 5)
    
    def get_coreset(self):
        return self.coreset
    
    def get_coreset_energy(self):
        return self.memory_energy
    
    def get_wasserstein_dist_df(self):
        return self.wasserstein_dist_df
    
    def get_energy_dist_df(self):
        return self.energy_dist_df

    def __len__(self):
        return len(self.coreset)
    
@torch.no_grad()
def accumulate_candidate(memory_option, model, energy, memory_x, memory_y, memory_energy, joint_targets, y_ans_idx, x, y):
        memory_x = torch.cat((memory_x, x.detach().cpu())) 
        memory_y = torch.cat((memory_y, y.detach().cpu()))
        if memory_option == 'confused_pred':
            true_index = (y - min(y)).type(torch.LongTensor).view(-1, 1).detach().cpu()
            energy_cpu = energy.detach().cpu()
            true_energy = torch.gather(energy_cpu, dim=1, index=true_index)
            other_energy = energy_cpu[energy_cpu != true_energy].view(energy_cpu.shape[0], -1)
            neg_energy = torch.min(other_energy, dim=1, keepdim=True)[0]
            memory_energy = torch.cat((memory_energy, true_energy-neg_energy))
        else:
            energy = model(x, joint_targets)
            true_energy = energy.gather(dim=1, index=y_ans_idx).detach().cpu()
            memory_energy = torch.cat((memory_energy, true_energy)) # |bx1|    
        return [memory_x.detach().cpu(), memory_y.detach().cpu(), memory_energy.detach().cpu()]

@torch.no_grad()
def online_bin_based_sampling(model, memory_size, class_per_tasks, candidates, task_class_set, x, y, device):
    x = torch.cat((candidates[0].long().to(device), x))
    y = torch.cat((candidates[1].long().to(device), y))
    joint_targets = get_target(task_class_set, y).to(device).long()
    y_ans_idx = get_ans_idx(task_class_set, y)
    cur_energy = model(x, joint_targets).detach().cpu()
    cur_energy = cur_energy.gather(dim=1, index=y_ans_idx)
    flatten_energy = cur_energy.view(-1) # |C_t|
    _, bin_idx = torch.sort(flatten_energy)
    bins = torch.linspace(0, len(bin_idx), memory_size).long()
    bins[-1] = bins[-1]-1
    memory_x = x[bin_idx][bins]
    memory_y = y[bin_idx][bins]
    mem_energy = cur_energy[bin_idx][bins].view(-1)
    return (memory_x.detach().cpu(), memory_y.detach().cpu()),  mem_energy   
    
def reservoir_sampling():
    None
    
    
def add_online_coreset():
    pass