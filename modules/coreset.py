
import os
import yaml
import torch
from torch.utils.data import ConcatDataset
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, energy_distance
from dataset import Coreset_Dataset

class Coreset_Manager(object):
    def __init__(self, model, args, memory, num_classes_per_tasks, memory_over_tasks_dataset, device):
        self.memory_energy, self.coresets = self._generate_and_evaluate_coreset(args, model, device, memory_over_tasks_dataset)
        self.available_memory_options = ['bin_based', 'random_sample', 'low_energy', 'high_energy', 'min_score', 'max_score', 'confused_pred']
        self.classes = self._get_class_name(args)
        self.wasserstein_dist_df = pd.DataFrame(index=self.available_memory_options, columns=self.classes)
        self.energy_dist_df = pd.DataFrame(index=self.available_memory_options, columns=self.classes)
        
        self.x, self.y, self.energy = memory
        self.num_classes_per_tasks = num_classes_per_tasks
        self.cl_list = list(set(list(self.y.numpy()))).sort()
    
    def _get_class_name(self, args):
        labels = []
        with open(os.path.join('utils','labels.yml')) as outfile:
            label_map = yaml.safe_load(outfile)
        if args.dataset == 'cifar10':
            cifar_label_map = label_map['CIFAR10_LABELS']
        else:
            raise NotImplemented('Only Available for Cifar10..... others will be implemented soon.....')
        cifar_labels = [i for i in range(args.num_classes)]
        return [list(cifar_label_map.keys())[list(cifar_label_map.values()).index(l)]\
                for l in cifar_labels]
            
    def _generate_and_evaluate_coreset(self, args, model, device, memory_over_tasks_dataset):
        memory_energy = torch.empty(0)
        for cur_class, cl in enumerate(self.cl_list[-1*self.num_classes_per_tasks]):
            index = (self.y == torch.tensor(cl)).nonzero(as_tuple=True)[0]
            cur_x = self.x[index]
            cur_y = self.y[index]
            cur_energy = self.energy[index]
            # memory_energy = torch.empty(0) # 이걸 왜 여기서 선언하지?
            if args.memory_option == "random_sample":
                memory_energy, memory_over_tasks_dataset = self._random_sample(args, cur_x, cur_y, cur_energy, memory_energy, memory_over_tasks_dataset)
            elif args.memory_option == "min_score":
                memory_energy, memory_over_tasks_dataset = self._score_based(args, model, device, cur_x, cur_y, cur_energy, memory_energy, memory_over_tasks_dataset, False)
            elif args.memory_option == "max_score":
                memory_energy, memory_over_tasks_dataset = self._score_based(args, model, device, cur_x, cur_y, cur_energy, memory_energy, memory_over_tasks_dataset, True)
            elif args.memory_option == "low_energy":
                memory_energy, memory_over_tasks_dataset = self._energy_based(args, cur_x, cur_y, cur_energy, memory_energy, memory_over_tasks_dataset, False)
            elif args.memory_option == "high_energy":
                memory_energy, memory_over_tasks_dataset = self._energy_based(args, cur_x, cur_y, cur_energy, memory_energy, memory_over_tasks_dataset, True)
            elif args.memory_option == "confused_pred":
                memory_energy, memory_over_tasks_dataset = self._confused_pred(args, cur_x, cur_y, cur_energy, memory_energy, memory_over_tasks_dataset)
            elif args.memory_option == "bin_based":
                memory_energy, memory_over_tasks_dataset = self._bin_based(args, cur_x, cur_y, cur_energy, memory_energy, memory_over_tasks_dataset)
            elif args.memory_option == "representation_based":
                '''
                Not Implemented yet
                '''
                # return self._representation_based()
                raise NotImplementedError("Not Implemented yet")
            elif args.memory_option == "ensemble":
                '''
                Not Implemented yet
                '''
                # return self._ensemble()
                raise NotImplementedError("Not Implemented yet")
            else:
                raise NotImplementedError("Not a valid option for coreset selection")
            self.wasserstein_dist_df[cur_class][args.memory_option] = self._calculate_wasserstein_distance(cur_energy, memory_energy)
            self.energy_dist_df[cur_class][args.memory_option] = self._calulate_energy_distance(cur_energy, memory_energy)
        return memory_energy, memory_over_tasks_dataset
    
    def _random_sample(self, args, cur_x, cur_y, cur_energy, memory_energy, memory_over_tasks_dataset):
        idx = torch.randperm(len(cur_x))[:args.memory_size]
        memory_x = cur_x[idx]
        memory_y = cur_y[idx]
        mem_energy = cur_energy[idx]
        memory_energy = torch.cat((memory_energy, mem_energy.detach().cpu().view(-1)))
        memory_dataset = Coreset_Dataset(memory_x, memory_y)
        memory_over_tasks_dataset = ConcatDataset([memory_over_tasks_dataset, memory_dataset])
        return memory_energy, memory_over_tasks_dataset
    
    def _energy_based(self, args, cur_x, cur_y, cur_energy, memory_energy, memory_over_tasks_dataset, energy_mode=True):
        '''
        energy_mode = True if you select coreset by high_energy
        else False
        '''
        idx = torch.topk(cur_energy, args.memory_size, dim=0, largest=energy_mode)[1]
        memory_x = cur_x[idx].view(args.img_size)
        memory_y = cur_y[idx]
        memory_dataset = Coreset_Dataset(memory_x, memory_y)
        mem_energy = cur_energy[idx]
        memory_energy = torch.cat((memory_energy, mem_energy.detach().cpu().view(-1)))
        memory_over_tasks_dataset = ConcatDataset([memory_over_tasks_dataset, memory_dataset])
        return memory_energy, memory_over_tasks_dataset
    
    def _calculate_score(args, model, device, cur_x, cur_y):
        score_tensor = torch.empty(0)
        for i in range(len(cur_x)):
            tmp_x = cur_x[i,:,:,:].view(args.img_size).to(device)
            tmp_y = cur_y[i].view(-1).long().to(device)
            tmp_x.require_grad = True
            e = model(tmp_x, tmp_y)
            e.backward
            score = tmp_x.grad
            sum = torch.abs(torch.mean(score)) + torch.var(score)
            score_tensor = torch.cat((score_tensor, sum.detach().cpu().view(-1)))
        return score_tensor
    
    def _score_based(self, args, model, device, cur_x, cur_y, cur_energy, memory_energy, memory_over_tasks_dataset, score_mode=True):
        '''
        score_mode = True if you select coreset by max_score
        else False
        '''
        score_tensor = self._calculate_score(args, model, device, cur_x, cur_y)
        idx = torch.topk(score_tensor, args.memory_size, dim=0, largest=score_mode)[1]
        memory_x = cur_x[idx]
        memory_y = cur_y[idx]
        mem_energy = cur_energy[idx]
        memory_energy = torch.cat((memory_energy, mem_energy.detach().cpu().view(-1)))
        memory_dataset = Coreset_Dataset(memory_x, memory_y)
        memory_over_tasks_dataset = ConcatDataset([memory_over_tasks_dataset, memory_dataset])
        return memory_energy, memory_over_tasks_dataset
    
    def _confused_pred(self, args, cur_x, cur_y, cur_energy, memory_energy, memory_over_tasks_dataset):
        idx = torch.topk(cur_energy, args.memory_size, dim=0, largest=True)[1]
        memory_x = cur_x[idx].view(args.img_size)
        memory_y = cur_y[idx]
        mem_energy = cur_energy[idx]
        memory_energy = torch.cat((memory_energy, mem_energy.detach().cpu().view(-1)))
        memory_dataset = Coreset_Dataset(memory_x, memory_y)
        memory_over_tasks_dataset = ConcatDataset([memory_over_tasks_dataset, memory_dataset])
        return memory_energy, memory_over_tasks_dataset
    
    def _bin_based(self, args, cur_x, cur_y, cur_energy, memory_energy, memory_over_tasks_dataset):
        flatten_energy = cur_energy.view(-1)
        _, bin_idx = torch.sort(flatten_energy)
        bins = torch.linspace(0, len(bin_idx), args.memory_size).int()
        bins[-1] = bins[-1]-1
        memory_x = cur_x[bin_idx][bins]
        memory_y = cur_y[bin_idx][bins]
        mem_energy = cur_energy[bin_idx][bins]
        memory_energy = torch.cat((memory_energy, mem_energy.detach().cpu().view(-1)))
        memory_dataset = Coreset_Dataset(memory_x, memory_y)
        memory_over_tasks_dataset = ConcatDataset([memory_over_tasks_dataset, memory_dataset])
        return memory_energy, memory_over_tasks_dataset
    
    def _representation_based(self):
        pass
    
    def _ensemble(self):
        pass
    
    def _calculate_wasserstein_distance(self, cur_energy, memory_energy):
        return np.round(wasserstein_distance(cur_energy.cpu().numpy().flatten(), memory_energy.cpu().numpy().flatten()), 5)
    
    def _calulate_energy_distance(self, cur_energy, memory_energy):
        return np.round(energy_distance(cur_energy.cpu().numpy().flatten(), memory_energy.cpu().numpy().flatten()), 5)
    
    def get_memory(self):
        return self.coresets
    
    def get_wasserstein_dist_df(self):
        return self.wasserstein_dist_df
    
    def get_energy_dist_df(self):
        return self.energy_dist_df
    