
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, energy_distance
from modules.dataset import Coreset_Dataset
from utils.utils import get_target, get_ans_idx


class Memory(nn.Module):
    def __init__(self, args, fix_slot_size=False):
        super().__init__()
        self.args = args
        self.fix_slot_size     = fix_slot_size
        self.memory_size       = args.memory_size
        self.memory_batch_size = args.batch_size
        
        print(f"Total Size of Buffer : {self.memory_size}")
        self.flattened_shape  = args.img_size*args.img_size*args.num_channels
        self.task_id          = 1 # 현재 training 중인 task (task 시작 전 update)
        self.num_cls_per_task = args.num_classes // args.num_tasks
        self.cur_memory_size  = self.memory_size // (self.num_cls_per_task*self.task_id) # 현재 저장할 class당 memory 크기
        
        self.memory_x      = torch.FloatTensor(self.memory_size, self.flattened_shape).fill_(0)
        self.memory_y      = torch.LongTensor(self.memory_size).fill_(0)
        self.memory_energy = torch.FloatTensor(self.memory_size).fill_(0)
        self.memory_rep    = torch.FloatTensor(self.memory_size, 512).fill_(0)
        self.mem_full_en   = torch.FloatTensor(self.memory_size, args.num_classes).fill_(0)
        
        self.new_x       = [torch.empty(0) for _ in range(self.num_cls_per_task)]
        self.new_y       = [torch.empty(0) for _ in range(self.num_cls_per_task)]
        self.new_energy  = [torch.empty(0) for _ in range(self.num_cls_per_task)]
        self.new_rep     = [torch.empty(0) for _ in range(self.num_cls_per_task)]
        self.new_full_en = [torch.empty(0) for _ in range(self.num_cls_per_task)]
    @property
    def x(self):
        return self.memory_x
    
    @property
    def y(self):
        return self.memory_y
    
    @property
    def energy(self):
        return self.memory_energy
    
    @property
    def rep(self):
        return self.memory_rep
    @property
    def full_energy(self):
        return self.mem_full_en
    
    def sample(self):
        return Coreset_Dataset(self.memory_x.view(-1, self.args.num_channels, self.args.img_size, self.args.img_size), \
            self.memory_y, self.memory_rep, self.mem_full_en)
    
    def set_task_id(self, t):
        self.task_id = t
        
    def set_cur_memory_size(self):
        self.cur_memory_size = self.memory_size // (self.num_cls_per_task*self.task_id)
        
    def _drop_samples(self):
        # memory size 업데이트 후 기존 memory에서 일부 sample drop하기 ex. task 1 -> task 2 : (100, 100) -> (50, 50)
        # memory 버퍼에 bin sampling을 적용  
        classes_in_memory  = (self.task_id-2) * self.num_cls_per_task
        former_memory_size = self.memory_size // classes_in_memory
        cur_memory_size    = self.memory_size // ((self.task_id-1)*self.num_cls_per_task)
        for i in range(classes_in_memory):
            memory_x_before        = self.memory_x[i*former_memory_size      : (i+1)*former_memory_size]
            memory_y_before        = self.memory_y[i*former_memory_size      : (i+1)*former_memory_size]
            memory_energy_before   = self.memory_energy[i*former_memory_size : (i+1)*former_memory_size]
            mem_energy_full_before = self.mem_full_en[i*former_memory_size   : (i+1)*former_memory_size]
            memory_rep_before      = self.memory_rep[i*former_memory_size    : (i+1)*former_memory_size]
            _, bin_idx = torch.sort(memory_energy_before)
            bins       = torch.linspace(0, len(bin_idx), cur_memory_size).long()
            
            bins[-1]   = bins[-1]-1
            self.memory_x[i*cur_memory_size      : (i+1)*cur_memory_size]      = memory_x_before[bin_idx][bins]
            self.memory_y[i*cur_memory_size      : (i+1)*cur_memory_size]      = memory_y_before[bin_idx][bins]
            self.memory_energy[i*cur_memory_size : (i+1)*cur_memory_size]      = memory_energy_before[bin_idx][bins]
            self.mem_full_en[i*cur_memory_size   : (i+1)*cur_memory_size]      = mem_energy_full_before[bin_idx][bins]
            self.memory_rep[i*cur_memory_size    : (i+1)*cur_memory_size]      = memory_rep_before[bin_idx][bins]
            
    def merge_samples(self):
        if self.task_id > 2:
            former_classes_in_memory = (self.task_id-2) * self.num_cls_per_task # 4
            classes_in_memory        = (self.task_id-1) * self.num_cls_per_task # 새롭게 sampling한 것들까지 포함 6
            former_memory_size       = self.memory_size // classes_in_memory # drop 이후 new_.. 을 제외한 이전까지 저장되어있던 메모리의 크기
            
            all_new_x       = torch.cat(self.new_x, dim=0)
            all_new_y       = torch.cat(self.new_y, dim=0)
            all_new_energy  = torch.cat(self.new_energy, dim=0)
            all_new_full_en = torch.cat(self.new_full_en, dim=0)
            all_new_rep     = torch.cat(self.new_rep, dim=0)
            
            self.memory_x[former_memory_size*former_classes_in_memory:former_memory_size*classes_in_memory]      = all_new_x
            self.memory_y[former_memory_size*former_classes_in_memory:former_memory_size*classes_in_memory]      = all_new_y
            self.memory_energy[former_memory_size*former_classes_in_memory:former_memory_size*classes_in_memory] = all_new_energy
            self.mem_full_en[former_memory_size*former_classes_in_memory:former_memory_size*classes_in_memory]   = all_new_full_en
            self.memory_rep[former_memory_size*former_classes_in_memory:former_memory_size*classes_in_memory]    = all_new_rep
           
            # reset candidates
            self.new_x       = [torch.empty(0) for _ in range(self.num_cls_per_task)]
            self.new_y       = [torch.empty(0) for _ in range(self.num_cls_per_task)]
            self.new_energy  = [torch.empty(0) for _ in range(self.num_cls_per_task)]
            self.new_full_en = [torch.empty(0) for _ in range(self.num_cls_per_task)]
            self.new_rep     = [torch.empty(0) for _ in range(self.num_cls_per_task)]
        else:
            all_new_x       = torch.cat(self.new_x, dim=0)
            all_new_y       = torch.cat(self.new_y, dim=0)
            all_new_energy  = torch.cat(self.new_energy, dim=0)
            all_new_full_en = torch.cat(self.new_full_en, dim=0)
            all_new_rep     = torch.cat(self.new_rep, dim=0)
            
            self.memory_x[:]      = all_new_x
            self.memory_y[:]      = all_new_y
            self.memory_energy[:] = all_new_energy
            self.mem_full_en[:]   = all_new_full_en
            self.memory_rep[:]    = all_new_rep
        
            # reset candidates
            self.new_x       = [torch.empty(0) for _ in range(self.num_cls_per_task)]
            self.new_y       = [torch.empty(0) for _ in range(self.num_cls_per_task)]
            self.new_energy  = [torch.empty(0) for _ in range(self.num_cls_per_task)]
            self.new_full_en = [torch.empty(0) for _ in range(self.num_cls_per_task)]
            self.new_rep     = [torch.empty(0) for _ in range(self.num_cls_per_task)]
        # torch.save(self.memory_x, f"memory_x_{self.task_id}.pt")
        # torch.save(self.memory_y, f"memory_y_{self.task_id}.pt")
    def update_memory(self, task_id):
        # 다음 task 시작 직전에 task_id 갱신
        # class당 memory 크기 갱신
        # memory size 업데이트 후 기존 memory에서 일부 sample drop하기 ex. task 1 -> task 2 : (100, 100) -> (50, 50)
        # 그 다음으로 memory_...와 new_... concat 해줌으로써 memory update 최종 완료
        
        self.set_task_id(task_id)
        self.set_cur_memory_size()
        self._drop_samples()
        self.merge_samples()
      
    def add_sample_online(self, x, y, energy, full_energy, reps, cur_cls_idx):
        # 매 iteration마다 수행
        '''
        Online Bin-Based Sampling with non-fixed memory slot
        
        cur_cls_idx : cls idx of current task class set.  ex) if num_cls_per_task = 2, then cur_cls_idx is 0 or 1
        x           : x with cur_cls_idx.
        y           : y with cur_cls_idx.
        '''
        
        x = x.view(-1, self.flattened_shape)
        # 남은 slot 크기보다 현재 batch 크기가 더 작다면
        # 그대로 concat
        self.new_x[cur_cls_idx]       = torch.cat((self.new_x[cur_cls_idx], x), dim=0)
        self.new_y[cur_cls_idx]       = torch.cat((self.new_y[cur_cls_idx], y), dim=0)
        self.new_energy[cur_cls_idx]  = torch.cat((self.new_energy[cur_cls_idx], energy.view(-1)), dim=0)
        self.new_rep[cur_cls_idx]     = torch.cat((self.new_rep[cur_cls_idx], reps), dim=0)
        self.new_full_en[cur_cls_idx] = torch.cat((self.new_full_en[cur_cls_idx], full_energy), dim=0)
        
        if self.new_x[cur_cls_idx].size(0) > self.cur_memory_size: 
            
            # 현재 저장된 memory의 크기가 class당 최대 메모리 크기보다 크다면
            # -> bin sampling
            _, bin_idx = torch.sort(self.new_energy[cur_cls_idx])
            bins       = torch.linspace(0, len(bin_idx), self.cur_memory_size).long()
            bins[-1]   = bins[-1]-1
            self.new_x[cur_cls_idx]       = self.new_x[cur_cls_idx][bin_idx][bins]
            self.new_y[cur_cls_idx]       = self.new_y[cur_cls_idx][bin_idx][bins]
            self.new_energy[cur_cls_idx]  = self.new_energy[cur_cls_idx][bin_idx][bins]
            self.new_rep[cur_cls_idx]     = self.new_rep[cur_cls_idx][bin_idx][bins]
            self.new_full_en[cur_cls_idx] = self.new_full_en[cur_cls_idx][bin_idx][bins]
            
    def _bin_based_sampling_offline(self, cur_cls):
        cur_energy = self.new_energy[cur_cls]
        _, bin_idx = torch.sort(cur_energy)
        bins       = torch.linspace(0, len(bin_idx), self.cur_memory_size).long()
        bins[-1]   = bins[-1]-1
        
        self.new_x[cur_cls]       = self.new_x[cur_cls][bin_idx][bins]
        self.new_y[cur_cls]       = self.new_y[cur_cls][bin_idx][bins]
        self.new_energy[cur_cls]  = self.new_energy[cur_cls][bin_idx][bins]
        self.new_rep[cur_cls]     = self.new_rep[cur_cls][bin_idx][bins]
        self.new_full_en[cur_cls] = self.new_full_en[cur_cls][bin_idx][bins]
        # self.memory_x[cur_cls*self.cur_memory_size :      (cur_cls+1)*self.cur_memory_size]      = temp_memory_x
        # self.memory_y[cur_cls*self.cur_memory_size :      (cur_cls+1)*self.cur_memory_size]      = temp_memory_y
        # self.memory_energy[cur_cls*self.cur_memory_size : (cur_cls+1)*self.cur_memory_size]      = temp_memory_energy
        
    @torch.no_grad()
    def add_sample_offline(self, loader, task_class_set, model, device):
        '''
        Calculate Energies
        '''
        cur_task_classes   = sorted(list(task_class_set)[-1*self.num_cls_per_task:])
        temp_memory_x      = torch.empty(0)
        temp_memory_y      = torch.empty(0)
        temp_memory_energy = torch.empty(0)
        temp_memory_rep    = torch.empty(0)
        temp_mem_full_en   = torch.empty(0)
        
        pbar = tqdm(loader, total=loader.__len__(), position=0, leave=True, desc="Calculating Energies of Task data", colour='red')
        
        for sample in pbar:
            x, y               = sample[0].to(device), sample[1].to(device)
            joint_targets      = get_target(task_class_set, y).to(device).long()
            y_ans_idx          = get_ans_idx(task_class_set, y).to(device)
            energy, rep        = model(x, joint_targets)
            true_energy        = energy.gather(dim=1, index=y_ans_idx) # |bs x 1| -> output with true class
            # true_rep           = rep.gather(dim=1, index=y_ans_idx)
            
            temp_memory_x       = torch.cat((temp_memory_x, x.detach().cpu()))
            temp_memory_y       = torch.cat((temp_memory_y, y.detach().cpu()))
            temp_memory_energy  = torch.cat((temp_memory_energy, true_energy.detach().cpu().view(-1)))
            temp_memory_rep     = torch.cat((temp_memory_rep, rep.detach().cpu()))
            temp_mem_full_en    = torch.cat((temp_mem_full_en, energy.detach().cpu()))
        
        for cur_cls_idx, cl in enumerate(cur_task_classes):        
            index                        = (temp_memory_y == cl).nonzero(as_tuple=True)[0]
            torch.save(temp_memory_x[index].view(-1, self.flattened_shape), f"asset/cls_full/cls_{cl}_full_x.pt")
            torch.save(temp_memory_y[index], f"asset/cls_full/cls_{cl}_full_y.pt")
            torch.save(temp_memory_energy[index], f"asset/cls_full/cls_{cl}_full_energy.pt")
            torch.save(temp_memory_rep[index,cl,:], f"asset/cls_full/cls_{cl}_full_rep.pt")
            
            self.new_x[cur_cls_idx]       = torch.cat((self.new_x[cur_cls_idx], temp_memory_x[index].view(-1, self.flattened_shape)), dim=0)
            self.new_y[cur_cls_idx]       = torch.cat((self.new_y[cur_cls_idx], temp_memory_y[index]), dim=0)
            self.new_energy[cur_cls_idx]  = torch.cat((self.new_energy[cur_cls_idx], temp_memory_energy[index]), dim=0)
            self.new_rep[cur_cls_idx]     = torch.cat((self.new_rep[cur_cls_idx], temp_memory_rep[index,cl,:]), dim=0)
            self.new_full_en[cur_cls_idx] = torch.cat((self.new_full_en[cur_cls_idx], temp_mem_full_en[index]), dim=0)
            self.new_full_en[cur_cls_idx] = torch.cat((self.new_full_en[cur_cls_idx], torch.zeros((self.new_full_en[cur_cls_idx].size(0), self.args.num_classes-(self.task_id)*len(cur_task_classes)))), dim=1)
        # for cur_cls_idx, cl in enumerate(cur_task_classes):
            self._bin_based_sampling_offline(cur_cls_idx)
    
    # def sample(self):
    #     indices = torch.from_numpy(np.random.choice(self.memory_x.size(0), self.memory_batch_size, replace=False))
    #     return (self.memory_x[indices].view(len(indices), self.args.num_channels, self.args.img_size, self.args.img_size), self.memory_y[indices])