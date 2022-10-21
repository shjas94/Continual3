
import os
import yaml
from tqdm import tqdm
import torch
from torch.utils.data import ConcatDataset
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, energy_distance
from modules.dataset import Coreset_Dataset

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
        self.memory_energy = None
                
    def add_coreset(self, model, device, memory, augmentation=None):
        x, y, energy = memory
        cl_list = sorted(list(set(list(y.numpy()))))
        
        for cur_class, cl in tqdm(enumerate(cl_list), desc="Getting Coreset"):
            index = (y == torch.tensor(cl)).nonzero(as_tuple=True)
            cur_x = x[index]
            cur_y = y[index]
            cur_energy = energy[index]

            if self.args.memory_option == "random_sample":
                self.memory_energy = self._random_sample(cur_x, cur_y, cur_energy)
                
            elif self.args.memory_option == "min_score":
                self.memory_energy = self._score_based(model, device, cur_x, cur_y, cur_energy, False)
                
            elif self.args.memory_option == "max_score":
                self.memory_energy = self._score_based(model, device, cur_x, cur_y, cur_energy, True)
                
            elif self.args.memory_option == "low_energy":
                self.memory_energy = self._energy_based(cur_x, cur_y, cur_energy, False)
                
            elif self.args.memory_option == "high_energy":
                self.memory_energy = self._energy_based(cur_x, cur_y, cur_energy, True)
                
            elif self.args.memory_option == "confused_pred":
                self.memory_energy = self._confused_pred(cur_x, cur_y, cur_energy)
                
            elif self.args.memory_option == "bin_based":
                self.memory_energy = self._bin_based(cur_x, cur_y, cur_energy)
                
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
            self.wasserstein_dist_df[self.classes[cur_class]][self.args.memory_option] = self._calculate_wasserstein_distance(cur_energy, self.memory_energy)
            self.energy_dist_df[self.classes[cur_class]][self.args.memory_option] = self._calulate_energy_distance(cur_energy, self.memory_energy)
    
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
        flatten_energy = cur_energy.view(-1)
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


# class Online_coreset_Manager(object):
#     def __init__(self, model, args, memory, num_classes_per_tasks, memory_over_tasks_dataset, device, memory_size, task_num, augmentation=None):
#         self.memory = torch.zeros((memory_size, args.img_size, args.img_size))
#         self.x, self.y, self.energy = memory
#         self.num_classes_per_tasks = num_classes_per_tasks
#         self.cl_list = sorted(list(set(list(self.y.numpy()))))
#         self.available_memory_options = ['bin_based', 'random_sample', 'low_energy', 'high_energy', 'min_score', 'max_score', 'confused_pred']
#         self.classes = self._get_class_name(args)
#         self.wasserstein_dist_df = pd.DataFrame(index=self.available_memory_options, columns=self.classes)
#         self.energy_dist_df = pd.DataFrame(index=self.available_memory_options, columns=self.classes)
#         self.memory_energy, self.coresets = self._generate_and_evaluate_coreset(args, model, device, memory_over_tasks_dataset, augmentation)
        
        
#     def _get_class_name(self, args):
#         labels = []
#         with open(os.path.join('utils','labels.yml')) as outfile:
#             label_map = yaml.safe_load(outfile)
#         if args.dataset == 'cifar10':
#             data_label_map = label_map['CIFAR10_LABELS']
#         elif args.dataset == 'tiny_imagenet':
#             data_label_map = label_map['TINYIMAGENET_LABELS']
#         else:
#             raise NotImplementedError('Only Available for Cifar10..... others will be implemented soon.....')
#         cifar_labels = [i for i in range(args.num_classes)]
#         return [list(data_label_map.keys())[list(data_label_map.values()).index(l)]\
#                 for l in cifar_labels]
            
#     def _generate_and_evaluate_coreset(self, args, model, device, memory_over_tasks_dataset, augmentation=None):
#         for cur_class, cl in tqdm(enumerate(self.cl_list[-self.num_classes_per_tasks:]), desc="Getting Coreset"):
#             index = (self.y == torch.tensor(cl)).nonzero(as_tuple=True)
#             cur_x = self.x[index]
#             cur_y = self.y[index]
#             cur_energy = self.energy[index]
#             # memory_energy = torch.empty(0) # ???
#             memory_energy = torch.empty(0)

#             if args.memory_option == "random_sample":
#                 memory_energy, memory_over_tasks_dataset = self._random_sample(args, cur_x, cur_y, cur_energy, memory_energy, memory_over_tasks_dataset)
#             elif args.memory_option == "min_score":
#                 memory_energy, memory_over_tasks_dataset = self._score_based(args, model, device, cur_x, cur_y, cur_energy, memory_energy, memory_over_tasks_dataset, False)
#             elif args.memory_option == "max_score":
#                 memory_energy, memory_over_tasks_dataset = self._score_based(args, model, device, cur_x, cur_y, cur_energy, memory_energy, memory_over_tasks_dataset, True)
#             elif args.memory_option == "low_energy":
#                 memory_energy, memory_over_tasks_dataset = self._energy_based(args, cur_x, cur_y, cur_energy, memory_energy, memory_over_tasks_dataset, False)
#             elif args.memory_option == "high_energy":
#                 memory_energy, memory_over_tasks_dataset = self._energy_based(args, cur_x, cur_y, cur_energy, memory_energy, memory_over_tasks_dataset, True)
#             elif args.memory_option == "confused_pred":
#                 memory_energy, memory_over_tasks_dataset = self._confused_pred(args, cur_x, cur_y, cur_energy, memory_energy, memory_over_tasks_dataset)
#             elif args.memory_option == "bin_based":
#                 memory_energy, memory_over_tasks_dataset = self._bin_based(args, cur_x, cur_y, cur_energy, memory_energy, memory_over_tasks_dataset)
#             elif args.memory_option == "representation_based":
#                 '''
#                 Not Implemented yet
#                 '''
#                 # return self._representation_based()
#                 raise NotImplementedError("Not Implemented yet")
#             elif args.memory_option == "ensemble":
#                 '''
#                 Not Implemented yet
#                 '''
#                 # return self._ensemble()
#                 raise NotImplementedError("Not Implemented yet")
#             else:
#                 raise NotImplementedError("Not a valid option for coreset selection")
#             self.wasserstein_dist_df[self.classes[cur_class]][args.memory_option] = self._calculate_wasserstein_distance(cur_energy, memory_energy)
#             self.energy_dist_df[self.classes[cur_class]][args.memory_option] = self._calulate_energy_distance(cur_energy, memory_energy)
#         return memory_energy, memory_over_tasks_dataset
    
#     def _random_sample(self, args, cur_x, cur_y, cur_energy, memory_energy, memory_over_tasks_dataset, augmentation=None):
#         idx = torch.randperm(len(cur_x))[:args.memory_size]
#         memory_x = cur_x[idx]
#         memory_y = cur_y[idx]
#         mem_energy = cur_energy[idx]
#         memory_energy = torch.cat((memory_energy, mem_energy.detach().cpu().view(-1)))
#         memory_dataset = Coreset_Dataset(memory_x, memory_y, transform=augmentation)
#         memory_over_tasks_dataset = ConcatDataset([memory_over_tasks_dataset, memory_dataset])
#         return memory_energy, memory_over_tasks_dataset
    
#     def _energy_based(self, args, cur_x, cur_y, cur_energy, memory_energy, memory_over_tasks_dataset, energy_mode=True, augmentation=None):
#         '''
#         energy_mode = True if you select coreset by high_energy
#         else False
#         '''
#         idx = torch.topk(cur_energy, args.memory_size, dim=0, largest=energy_mode)[1]
#         memory_x = cur_x[idx].view((-1, args.num_channels, args.img_size, args.img_size))
#         memory_y = cur_y[idx]
#         memory_dataset = Coreset_Dataset(memory_x, memory_y, transform=augmentation)
#         mem_energy = cur_energy[idx]
#         memory_energy = torch.cat((memory_energy, mem_energy.detach().cpu().view(-1)))
#         memory_over_tasks_dataset = ConcatDataset([memory_over_tasks_dataset, memory_dataset])
#         return memory_energy, memory_over_tasks_dataset
    
#     def _calculate_score(self, args, model, device, cur_x, cur_y):
#         score_tensor = torch.empty(0)
#         for i in range(len(cur_x)):
#             tmp_x = cur_x[i,:,:,:].view((-1, args.num_channels, args.img_size, args.img_size)).to(device)
#             tmp_y = cur_y[i].view(-1).long().to(device)
#             tmp_x.requires_grad = True
#             e = model(tmp_x, tmp_y)[0]
#             e.backward()
#             score = tmp_x.grad
#             sum = torch.abs(torch.mean(score)) + torch.var(score)
#             score_tensor = torch.cat((score_tensor, sum.detach().cpu().view(-1)))
#         return score_tensor
    
#     def _score_based(self, args, model, device, cur_x, cur_y, cur_energy, memory_energy, memory_over_tasks_dataset, score_mode=True, augmentation=None):
#         '''
#         score_mode = True if you select coreset by max_score
#         else False
#         '''
#         score_tensor = self._calculate_score(args, model, device, cur_x, cur_y)
#         idx = torch.topk(score_tensor, args.memory_size, dim=0, largest=score_mode)[1]
#         memory_x = cur_x[idx]
#         memory_y = cur_y[idx]
#         mem_energy = cur_energy[idx]
#         memory_energy = torch.cat((memory_energy, mem_energy.detach().cpu().view(-1)))
#         memory_dataset = Coreset_Dataset(memory_x, memory_y, transform=augmentation)
#         memory_over_tasks_dataset = ConcatDataset([memory_over_tasks_dataset, memory_dataset])
#         return memory_energy, memory_over_tasks_dataset
    
#     def _confused_pred(self, args, cur_x, cur_y, cur_energy, memory_energy, memory_over_tasks_dataset, augmentation=None):
#         idx = torch.topk(cur_energy, args.memory_size, dim=0, largest=True)[1]
#         memory_x = cur_x[idx].view(args.img_size)
#         memory_y = cur_y[idx]
#         mem_energy = cur_energy[idx]
#         memory_energy = torch.cat((memory_energy, mem_energy.detach().cpu().view(-1)))
#         memory_dataset = Coreset_Dataset(memory_x, memory_y, transform=augmentation)
#         memory_over_tasks_dataset = ConcatDataset([memory_over_tasks_dataset, memory_dataset])
#         return memory_energy, memory_over_tasks_dataset
    
#     def _bin_based(self, args, cur_x, cur_y, cur_energy, memory_energy, memory_over_tasks_dataset, augmentation=None):
#         flatten_energy = cur_energy.view(-1)
#         _, bin_idx = torch.sort(flatten_energy)
#         bins = torch.linspace(0, len(bin_idx), args.memory_size).long()
#         bins[-1] = bins[-1]-1
#         memory_x = cur_x[bin_idx][bins]
#         memory_y = cur_y[bin_idx][bins]
#         mem_energy = cur_energy[bin_idx][bins]
#         memory_energy = torch.cat((memory_energy, mem_energy.detach().cpu().view(-1)))
#         memory_dataset = Coreset_Dataset(memory_x, memory_y, transform=augmentation)
#         memory_over_tasks_dataset = ConcatDataset([memory_over_tasks_dataset, memory_dataset])
#         return memory_energy, memory_over_tasks_dataset
    
#     def _representation_based(self):
#         pass
    
#     def _ensemble(self):
#         pass
    
#     def _calculate_wasserstein_distance(self, cur_energy, memory_energy):
#         return np.round(wasserstein_distance(cur_energy.cpu().numpy().flatten(), memory_energy.cpu().numpy().flatten()), 5)
    
#     def _calulate_energy_distance(self, cur_energy, memory_energy):
#         return np.round(energy_distance(cur_energy.cpu().numpy().flatten(), memory_energy.cpu().numpy().flatten()), 5)
    
#     def _reservoir_sampling(self):
#         pass
    
#     def get_coreset(self):
#         return self.coresets
    
#     def get_wasserstein_dist_df(self):
#         return self.wasserstein_dist_df
    
#     def get_energy_dist_df(self):
#         return self.energy_dist_df
